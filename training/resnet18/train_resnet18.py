import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Hyperparameters
num_epochs = 200
batch_size = 256
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets
train_dataset = datasets.ImageFolder(root='pa100kdata/train', transform=transform)
val_dataset = datasets.ImageFolder(root='pa100kdata/val', transform=transform)


# Calculate class weights for WeightedRandomSampler
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)

# Create sampler for balanced batches
sampler = WeightedRandomSampler(weights=class_weights[train_dataset.targets], num_samples=len(train_dataset), replacement=True)

# Data loaders
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# Soft F1 Loss Function
class SoftF1Loss(nn.Module):
    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        y_true = torch.eye(logits.shape[1]).to(device)[labels]  
        tp = (probs * y_true).sum(dim=1)
        fp = ((1 - y_true) * probs).sum(dim=1)
        fn = (y_true * (1 - probs)).sum(dim=1)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
        return 1 - f1.mean()

criterion = SoftF1Loss() # Maximaxe f1 score loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # AdamW optimizer

best_f1 = 0
best_model_wts = None

# File to save metrics
log_file = open('training_log.txt', 'w')
log_file.write('Epoch,Train Loss,Train F1,Train Recall,Train Precision,Train Top1 Acc,Val Loss,Val F1,Val Recall,Val Precision,Val Top1 Acc\n')

# Training and Validation Loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Training phase
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_top1_acc = correct / total

    print(f'Train Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} Top1 Acc: {epoch_top1_acc:.4f}')

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_preds = []
    val_labels = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
    val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
    val_epoch_top1_acc = val_correct / val_total

    print(f'Val Loss: {val_epoch_loss:.4f} F1: {val_epoch_f1:.4f} Recall: {val_epoch_recall:.4f} Precision: {val_epoch_precision:.4f} Top1 Acc: {val_epoch_top1_acc:.4f}')

    # Log metrics to file
    log_file.write(f'{epoch+1},{epoch_loss:.4f},{epoch_f1:.4f},{epoch_recall:.4f},{epoch_precision:.4f},{epoch_top1_acc:.4f},{val_epoch_loss:.4f},{val_epoch_f1:.4f},{val_epoch_recall:.4f},{val_epoch_precision:.4f},{val_epoch_top1_acc:.4f}\n')

    # Save the best model
    if val_epoch_f1 > best_f1:
        best_f1 = val_epoch_f1
        best_model_wts = model.state_dict()

# Save the final model checkpoint

# Save the best model checkpoint
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'resnet18_best.pth')
torch.save(model.state_dict(), 'resnet18_final.pth')
log_file.close()
print('Training complete.')
