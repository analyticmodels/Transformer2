def train_model(model, train_loader, val_loader, num_epochs=15, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move data to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Validate input shapes and values
                    if input_ids.max() >= model.vocab_size:
                        print(f"Warning: Batch {batch_idx} contains tokens >= vocab_size")
                        input_ids = torch.clamp(input_ids, 0, model.vocab_size - 1)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Update progress
                    train_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})
                    
                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    print(f"Input shapes - IDs: {input_ids.shape}, Mask: {attention_mask.shape}")
                    print(f"Input range - Min: {input_ids.min()}, Max: {input_ids.max()}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    if input_ids.max() >= model.vocab_size:
                        input_ids = torch.clamp(input_ids, 0, model.vocab_size - 1)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step()
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model.pt')
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