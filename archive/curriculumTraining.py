from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
import math
from tqdm import tqdm
import torch.cuda.amp as amp

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def get_curriculum_length(epoch, num_epochs, min_length=64, max_length=512):
    """Gradually increase sequence length during training"""
    return min(max_length, min_length + (max_length - min_length) * (epoch / (num_epochs // 2)))

def train_model(model, train_loader, val_loader, num_epochs=50, patience=5, 
                accumulation_steps=16, max_grad_norm=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,  # Initial learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    total_steps = (len(train_loader) // accumulation_steps) * num_epochs
    
    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,  # Peak learning rate
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of steps in warmup
        anneal_strategy='cos',
        cycle_momentum=True,
        div_factor=25.0,
        final_div_factor=10000.0,
        base_momentum=0.85,
        max_momentum=0.95
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.15)
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            curr_seq_length = int(get_curriculum_length(epoch, num_epochs))
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc=f"Epoch {epoch+1}/{num_epochs} (seq_len={curr_seq_length})")
            else:
                progress_bar = enumerate(train_loader)
            
            for batch_idx, batch in progress_bar:
                input_ids = batch['input_ids'][:, :curr_seq_length].to(device)
                attention_mask = batch['attention_mask'][:, :curr_seq_length].to(device)
                labels = batch['labels'][:, :curr_seq_length].to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Step the scheduler after each batch
                    scheduler.step()
                
                total_loss += loss.item() * accumulation_steps
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            'loss': f"{total_loss/(batch_idx+1):.4f}",
                            'lr': f"{current_lr:.2e}",
                            'seq_len': curr_seq_length
                        })
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'][:, :curr_seq_length].to(device)
                    attention_mask = batch['attention_mask'][:, :curr_seq_length].to(device)
                    labels = batch['labels'][:, :curr_seq_length].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'\nEpoch {epoch+1}:')
                print(f'Average training loss: {avg_train_loss:.4f}')
                print(f'Average validation loss: {avg_val_loss:.4f}')
                print(f'Learning rate: {current_lr:.2e}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': best_val_loss,
                    'config': {
                        'd_model': model.d_model,
                        'num_heads': model.transformer_blocks[0].attention.num_heads,
                        'num_layers': len(model.transformer_blocks),
                        'd_ff': model.transformer_blocks[0].ff[0].out_features,
                    }
                }, 'best_model.pt')
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print("Saved new best model!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
                    
    except Exception as e:
        print(f"Training interrupted by error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return model