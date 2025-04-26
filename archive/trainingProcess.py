# Initialize model and training components
model = SmallTransformer(
    vocab_size=50000,
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_len=512
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train the model
train_model(model, train_dataloader, optimizer, scheduler, num_epochs=10)