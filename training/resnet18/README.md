### DataLoader

- **Data Transformations**: 
  Images are resized to 128x128 pixels, converted to tensors, and normalized.
- **Datasets**: 
  Training and validation datasets are loaded from directories using the ImageFolder structure.
- **Class Balancing**: 
  Class weights are calculated to handle class imbalance, and a `WeightedRandomSampler` is used to create balanced batches for the training data.
- **DataLoaders**: 
  DataLoaders are set up for both training and validation datasets, with the training DataLoader using the weighted sampler.

### Model

- **Model Architecture**: 
  A pre-trained ResNet-18 model is used.
- **Modification**: 
  The final fully connected layer is replaced to match the number of classes in the dataset.
- **Device Setup**: 
  The model is moved to the appropriate device (CPU or GPU).

### Loss

- **Custom Loss Function**: 
  A custom loss function (`SoftF1Loss`) is defined to maximize the F1 score. This function calculates the F1 score based on true positives, false positives, and false negatives and uses `1 - F1 score` as the loss.
- **Optimizer**: 
  The AdamW optimizer is used with a specified learning rate and weight decay.

The script trains the model over multiple epochs, logging performance metrics (loss, F1 score, recall, precision, accuracy) for both training and validation phases, and saves the best model based on validation F1 score.
