### DataLoader

- **Data Transformations**: 
  Images are resized to 128x128 pixels, converted to tensors, and normalized.
- **Datasets**: 
  Training and validation datasets are loaded from directories using the ImageFolder structure.
- **DataLoaders**: 
  DataLoaders are set up for both training and validation datasets, with the training DataLoader shuffling the data for each epoch.

### Model

- **Model Architecture**: 
  An EfficientNet-B0 model pre-trained on ImageNet is used.
- **Modification**: 
  The final fully connected layer is replaced with a linear layer having two output classes to match the number of classes in the dataset.
- **Device Setup**: 
  The model is moved to the appropriate device (CPU or GPU).

### Loss

- **Loss Function**: 
  Cross-Entropy Loss is used as the criterion for optimization.
- **Optimizer**: 
  The Adam optimizer is used with a specified learning rate.

The script trains the model over multiple epochs, logging performance metrics (loss, F1 score, recall, precision, accuracy) for both training and validation phases, and saves the best model based on validation F1 score.
