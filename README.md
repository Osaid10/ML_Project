# Advanced Deep Learning Techniques for Fashion-MNIST Classification

## Overview
This project explores advanced deep learning techniques to classify images from the Fashion-MNIST dataset. The Fashion-MNIST dataset consists of 70,000 grayscale images of 10 clothing categories, such as T-shirts, trousers, and shoes. By leveraging MobileNetV2, transfer learning, and state-of-the-art techniques like data augmentation and cyclical learning rate scheduling, the model achieves over 93% validation accuracy.

## Features
- **Dataset**: Fashion-MNIST with 28x28 grayscale images.
- **Model**: MobileNetV2 pretrained on ImageNet.
- **Techniques Used**:
  - Transfer Learning
  - Data Augmentation (Rotation, Zoom, Shear, etc.)
  - Cyclical Learning Rate Scheduling
  - Regularization (Dropout, L2 Penalties)
- **Performance**:
  - Training Accuracy: ~95%
  - Validation Accuracy: ~93%

## Project Structure
- `data/`: Contains dataset preprocessing scripts.
- `models/`: Contains model architecture and training scripts.
- `visualizations/`: Scripts to generate performance graphs, confusion matrices, and prediction confidence boxplots.
- `README.md`: Project overview and instructions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-mnist-classification.git
   cd fashion-mnist-classification
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training the Model
1. Run the training script:
   ```bash
   python train.py
   ```
2. Training configurations (e.g., batch size, learning rate) can be modified in `config.py`.

### Evaluating the Model
1. Evaluate the model on the test set:
   ```bash
   python evaluate.py
   ```
2. Generate performance visualizations:
   ```bash
   python visualize_results.py
   ```

### Model Deployment
Export the trained model:
```bash
python export_model.py
```
The model will be saved in the `exports/` directory as `fashion_mnist_model.h5`.

## Results
### Training and Validation Accuracy
![Accuracy Plot](visualizations/accuracy_plot.png)

### Training and Validation Loss
![Loss Plot](visualizations/loss_plot.png)

### Confusion Matrix
![Confusion Matrix](visualizations/confusion_matrix.png)

### Prediction Visualization
![Example Predictions](visualizations/example_predictions.png)

### Prediction Confidence Boxplot
![Confidence Boxplot](visualizations/prediction_probabilities_boxplot.png)

## Key Findings
- MobileNetV2 achieves high accuracy with minimal computational resources.
- Advanced data augmentation techniques improve model robustness.
- Cyclical learning rates stabilize training and enhance convergence.

## Future Work
- Explore other architectures like Vision Transformers.
- Apply the methodology to more complex datasets.
- Integrate explainability tools like Grad-CAM for better interpretability.

## References
1. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2021). Deep Residual Learning for Image Recognition.
