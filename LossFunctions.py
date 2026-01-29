
criterion = nn.CrossEntropyLoss()  # Handles logits directly (no sigmoid needed)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001
)

# Training loop (40 epochs, batch_size=8)
for epoch in range(40):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)  # Shape: [8, 3, 224, 224]
        labels = labels.to(device)  # Shape: [8] (integer labels: 0 or 1)

        optimizer.zero_grad()
        outputs = model(images)  # Shape: [8, 2] (logits)
        loss = criterion(outputs, labels)  # Cross-entropy loss
        loss.backward()
        optimizer.step()

    # Validation (compute AUC every epoch)
    model.eval()
    with torch.no_grad():
        y_true, y_scores = [], []
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(malignant)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(y_true, y_scores)
        print(f"Epoch {epoch + 1}, Val AUC: {val_auc:.5f}")