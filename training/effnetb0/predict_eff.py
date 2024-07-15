import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import shutil

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
])

# Load test dataset
test_dataset = datasets.ImageFolder(root='pa100kdata/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the best model
model = models.efficientnet_b0(pretrained=True)  # EfficientNet-B0 model
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # 2 sınıflı çıkış katmanı
model.load_state_dict(torch.load('effb0_best.pth'), strict=False)
model = model.to(device)
model.eval()

# Directory to save predictions
output_dir = 'predictions_effb0'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Class names
class_names = test_dataset.classes

# Predict and save results
with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        predicted_class = class_names[preds.item()]
        image_path = test_dataset.samples[i][0]
        image_name = os.path.basename(image_path)

        prediction_path = os.path.join(output_dir,
                                       f'{predicted_class}_{image_name}')
        Image.open(image_path).save(prediction_path)

        print(f'Image {image_name} predicted as {predicted_class}')

print(f'Predictions saved in {output_dir} directory.')
