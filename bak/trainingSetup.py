import torch
import torch.nn as nn
import math
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def initialize_weights(model):
    """More conservative weight initialization"""
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    model.apply(_init_weights)
    return model

def train_model(model, train_loader, val_loader, num_epochs=30, patience=3, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize weights
    model = initialize_weights(model)
    model = model.to(device)
    
    total_steps = (len(train_loader) // accumulation_steps) * num_epochs
    
    # Even more conservative optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,  # Further reduced learning rate
        weight_decay=0.001,  # Reduced weight decay
        eps=1e-9,  # Increased numerical stability
        betas=(0.9, 0.98)  # Modified beta values
    )
    
    # Longer warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 3,  # 33% warmup
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean', label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()  # Add mixed precision training
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            valid_batch_count = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Mixed precision forward pass
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\nNaN/Inf loss detected in batch {batch_idx}. Skipping batch.")
                        continue
                    
                    # Scale loss
                    loss = loss / accumulation_steps
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        # Unscale before gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                        
                        # Check gradients
                        valid_gradients = True
                        grad_norm = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    valid_gradients = False
                                    break
                        grad_norm = grad_norm ** 0.5
                        
                        if valid_gradients and grad_norm < 10:  # Add gradient norm check
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            train_loss += loss.item() * accumulation_steps
                            valid_batch_count += 1
                        else:
                            print(f"\nSkipping update - Invalid gradients or grad norm: {grad_norm:.2f}")
                        
                        optimizer.zero_grad()
                    
                    # Update progress bar
                    current_lr = scheduler.get_last_lr()[0]
                    if valid_batch_count > 0:
                        avg_loss = train_loss / valid_batch_count
                    else:
                        avg_loss = float('nan')
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'valid_batches': valid_batch_count
                    })
                    
                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    continue
            
            if valid_batch_count > 0:
                avg_train_loss = train_loss / valid_batch_count
                print(f"\nEpoch {epoch+1} completed with {valid_batch_count} valid batches")
                print(f"Average training loss: {avg_train_loss:.4f}")
                
                # Only do validation if we had valid training batches
                model.eval()
                val_loss = 0
                val_batch_count = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            val_loss += loss.item()
                            val_batch_count += 1
                
                if val_batch_count > 0:
                    avg_val_loss = val_loss / val_batch_count
                    print(f'Average validation loss: {avg_val_loss:.4f}')
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': best_val_loss,
                        }, 'best_model.pt')
                        print("Saved new best model!")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f'Early stopping triggered after {epoch+1} epochs')
                            break
            else:
                print(f"\nEpoch {epoch+1} had no valid batches - considering architectural changes")
                
    except Exception as e:
        print(f"Training interrupted by error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return model