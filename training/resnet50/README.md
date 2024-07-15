### DataLoader

- **Data Transformations**: 
  Images are resized to 128x128 pixels, converted to tensors, and normalized using mean and standard deviation values commonly used for ImageNet.
- **Datasets**: 
  Training and validation datasets are loaded from directories using the ImageFolder structure.
- **DataLoaders**: 
  DataLoaders are set up for both training and validation datasets, with the training DataLoader shuffling the data for each epoch.

### Model

- **Model Architecture**: 
  A pre-trained ResNet50 model is used.
- **Modification**: 
  The final fully connected layer is replaced with a linear layer having the same number of output units as the number of classes in the dataset.
- **Device Setup**: 
  The model is moved to the appropriate device (CPU or GPU).

### Loss

- **Loss Function**: 
  A custom Soft F1 Loss function is defined, which computes the F1 score for each class and returns the mean value.
- **Optimizer**: 
  The Adam optimizer is used with a specified learning rate.

The script trains the model over multiple epochs, logging performance metrics (loss, F1 score, recall, precision, accuracy) for both training and validation phases, and saves the best model based on validation F1 score.
