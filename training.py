import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.cuda.amp as amp
import os
import argparse

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Creates a cosine learning rate scheduler with warmup.
    
    Args:
        optimizer: The optimizer to use
        num_warmup_steps: Number of steps for warmup
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def get_curriculum_length(epoch, num_epochs, min_length=64, max_length=512):
    """
    Gradually increase sequence length during training.
    
    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        min_length: Minimum sequence length to start with
        max_length: Maximum sequence length to end with
        
    Returns:
        int: Current sequence length for this epoch
    """
    # Reach max_length by the half-way point of training. max() guards the single-epoch
    # case, where num_epochs // 2 would be zero.
    ramp_epochs = max(1, num_epochs // 2)
    return min(max_length, min_length + (max_length - min_length) * (epoch / ramp_epochs))

def model_config(model):
    """
    Record the architecture of a model so inference can rebuild it.

    Every value is read off the model itself rather than off the training arguments,
    so the config always describes the weights it is saved next to.

    Args:
        model: The model being checkpointed

    Returns:
        dict: Architecture hyperparameters. Attributes the model does not expose are
            omitted rather than recorded as None, so that a reader's fallbacks still fire.
    """
    attributes = {
        'vocab_size': 'vocab_size',
        'd_model': 'd_model',
        'num_heads': 'num_heads',
        'num_layers': 'num_layers',
        'd_ff': 'd_ff',
        'max_seq_len': 'max_seq_len',
        'dropout': 'dropout_p',
    }
    config = {}
    for key, attribute in attributes.items():
        value = getattr(model, attribute, None)
        if value is not None:
            config[key] = value
    return config

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, loss,
                    train_losses, val_losses, extra_config=None):
    """
    Write a training checkpoint.

    Args:
        path: Destination file
        epoch: Epoch index being saved
        model, optimizer, scheduler, scaler: Training state to serialize
        loss: Validation loss associated with this checkpoint
        train_losses, val_losses: Loss history so far
        extra_config: Training settings to record alongside the architecture
    """
    config = model_config(model)
    if extra_config:
        config.update(extra_config)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
    }, path, pickle_protocol=5, _use_new_zipfile_serialization=True)

def sibling_checkpoint(checkpoint_path, name):
    """
    Place an auxiliary checkpoint beside the main one.

    Periodic, interrupted and failed checkpoints used to be written to the process
    working directory while only the best model honoured --output_dir, so a single
    run scattered its output across two locations.

    Args:
        checkpoint_path (str): Path the best model is written to
        name (str): Bare filename for the auxiliary checkpoint

    Returns:
        str: `name` resolved into the same directory as checkpoint_path
    """
    return os.path.join(os.path.dirname(checkpoint_path), name)


def train_model(model, train_loader, val_loader, num_epochs=50, patience=5,
                accumulation_steps=16, max_grad_norm=0.5, use_curriculum=False,
                checkpoint_path='best_model.pt', checkpoint_freq=5, optimizer_type='adamw',
                tokenizer_meta=None):
    """
    Train a transformer model with optional curriculum learning.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train for
        patience: Patience for early stopping
        accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum norm for gradient clipping
        use_curriculum: Whether to use curriculum learning (gradually increase sequence length)
        checkpoint_path: Path to save the best model
        checkpoint_freq: Frequency (in epochs) to save checkpoints
        tokenizer_meta: Provenance of the tokenizer, recorded in the checkpoint
        optimizer_type: One of 'adam', 'adamw' or 'sgd'

    Returns:
        tuple: (trained model, training losses, validation losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Calculate total optimizer steps. On a small dataset len(train_loader) can be below
    # accumulation_steps, which would otherwise leave total_steps at 0 -- that makes
    # OneCycleLR raise and collapses the MultiStepLR milestones onto step 0, where gamma
    # gets applied twice before the first batch.
    total_steps = max(1, (len(train_loader) // accumulation_steps) * num_epochs)
    
    # Setup optimizer based on chosen type
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print("Using Adam optimizer")
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001
        )
        print("Using SGD optimizer with momentum")
    else:  # default to AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        print("Using AdamW optimizer")
    
    # Learning rate scheduler
    # Set base criteria
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    
    # Configure scheduler based on training approach, then optimizer. Curriculum learning
    # is checked first so that --optimizer sgd --use_curriculum still gets the OneCycle
    # setup the max_lr below was written for.
    if use_curriculum:
        # OneCycleLR scheduler for curriculum learning
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1 if optimizer_type.lower() == 'sgd' else 5e-4,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=10000.0,
            base_momentum=0.85,
            max_momentum=0.95
        )
        print("Using OneCycleLR scheduler with curriculum learning")
        # Higher label smoothing for curriculum learning
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.15)
    elif optimizer_type.lower() == 'sgd':
        # For SGD, use a multi-step LR scheduler. Milestones must be distinct and past
        # step 0, or gamma is applied repeatedly before training starts.
        milestones = sorted({max(1, total_steps // 3), max(2, 2 * total_steps // 3)})
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
        print(f"Using MultiStepLR scheduler with milestones at steps {milestones}")
    else:
        # Cosine scheduler with warmup for standard training
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 5,  # 20% warmup
            num_training_steps=total_steps,
            num_cycles=0.5
        )
        print("Using cosine scheduler with warmup")
    
    # Setup for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training state tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    epoch = 0
    avg_val_loss = float('inf')

    # Training settings worth recording next to the architecture
    training_config = {
        'use_curriculum': use_curriculum,
        'optimizer_type': optimizer_type,
    }
    if tokenizer_meta:
        # Vocabulary size cannot identify a tokenizer -- every corpus here trains to
        # 50000 -- so the checkpoint records which one it was actually trained with.
        training_config.update({
            'tokenizer_dataset': tokenizer_meta.get('dataset'),
            'tokenizer_vocab_sha256': tokenizer_meta.get('vocab_sha256'),
        })

    print(f"Starting training on {device} with {'curriculum' if use_curriculum else 'standard'} learning")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            # Set sequence length for this epoch if using curriculum learning
            curr_seq_length = int(get_curriculum_length(epoch, num_epochs)) if use_curriculum else None
            
            # Create progress bar
            progress_desc = f"Epoch {epoch+1}/{num_epochs}"
            if use_curriculum:
                progress_desc += f" (seq_len={curr_seq_length})"
                
            progress_bar = tqdm(enumerate(train_loader), 
                              total=len(train_loader),
                              desc=progress_desc)
            
            # Training loop
            for batch_idx, batch in progress_bar:
                # Get batch data and move to device
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                # Apply curriculum learning if enabled
                if use_curriculum and curr_seq_length is not None:
                    input_ids = input_ids[:, :curr_seq_length]
                    attention_mask = attention_mask[:, :curr_seq_length]
                    labels = labels[:, :curr_seq_length]
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Optimizer and scheduler step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler._last_lr[0]
                progress_bar.set_postfix({
                    'loss': f"{total_loss/(batch_idx+1):.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    
                    # Apply curriculum learning if enabled
                    if use_curriculum and curr_seq_length is not None:
                        input_ids = input_ids[:, :curr_seq_length]
                        attention_mask = attention_mask[:, :curr_seq_length]
                        labels = labels[:, :curr_seq_length]
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            print(f'Learning rate: {current_lr:.2e}')
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print("Saving best model...")
                save_checkpoint(
                    checkpoint_path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, train_losses, val_losses, training_config
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(
                    sibling_checkpoint(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt'), epoch, model, optimizer, scheduler,
                    scaler, avg_val_loss, train_losses, val_losses, training_config
                )

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        save_checkpoint(
            sibling_checkpoint(checkpoint_path, 'interrupted_checkpoint.pt'), epoch, model, optimizer, scheduler, scaler,
            avg_val_loss, train_losses, val_losses, training_config
        )

    except Exception:
        # Do not swallow the failure: callers previously received a partially trained
        # model back and reported success. Save what there is, then let it propagate.
        print("Training failed. Saving a recovery checkpoint before re-raising...")
        try:
            save_checkpoint(
                sibling_checkpoint(checkpoint_path, 'failed_checkpoint.pt'), epoch, model, optimizer, scheduler, scaler,
                avg_val_loss, train_losses, val_losses, training_config
            )
        except Exception as save_error:
            print(f"Could not save recovery checkpoint: {save_error}")
        raise

    return model, train_losses, val_losses

def main():
    """Main function to run training from the command line."""
    parser = argparse.ArgumentParser(description='Train transformer model on text data')
    parser.add_argument('--dataset', type=str, default='reddit', choices=['reddit', 'bluesky', 'tinystories'],
                        help='Dataset to use for training (default: reddit)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for (default: 30)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping (default: 3)')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use (default: adamw)')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Use curriculum learning (gradually increase sequence length)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save model checkpoints')
    parser.add_argument('--use_ollama', action='store_true',
                        help='Not supported for training; use inference.py --use_ollama instead')
    # Model training parameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    args = parser.parse_args()

    if args.use_ollama:
        # Ollama serves generations over HTTP, so there are no parameters to optimize and
        # no gradients to backpropagate. It is an inference backend only.
        parser.error(
            "--use_ollama is not supported for training: an Ollama model is served over "
            "HTTP and exposes no trainable parameters or gradients. Use it for generation "
            "instead: python inference.py --use_ollama --ollama_model <name>"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training on {args.dataset} dataset")
    print(f"Model: d_model={args.d_model}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}, d_ff={args.d_ff}, dropout={args.dropout}")
    print(f"Using {args.optimizer} optimizer with {'curriculum' if args.use_curriculum else 'standard'} learning")

    # Import here to avoid circular imports
    from dataloader import get_dataloaders
    from paths import tokenizer_dir
    from tokenizer import read_tokenizer_meta

    # Load data
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type=args.dataset
    )

    # Create model
    from coreTransformer import SmallTransformer
    model = SmallTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout
    )

    # Train model
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        accumulation_steps=args.accumulation_steps,
        use_curriculum=args.use_curriculum,
        checkpoint_path=checkpoint_path,
        optimizer_type=args.optimizer,
        tokenizer_meta=read_tokenizer_meta(tokenizer_dir(args.dataset)),
    )

    print(f"\nTraining completed. Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()