import torch
import torch.nn as nn
import math
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=30, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Basic training setup with conservative values
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("Testing forward pass...")
    # Test forward pass with a single batch
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        print(f"Initial loss: {loss.item()}")
        break
    
    # If we get here without errors, start training
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'checkpoint.pt')
    
    return model