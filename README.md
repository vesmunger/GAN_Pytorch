## MNIST GAN with PyTorch Lightning

### Author: vesmunger

This project implements a Generative Adversarial Network (GAN) using PyTorch Lightning to generate handwritten digits similar to the MNIST dataset. The repository demonstrates building, training, and evaluating a GAN with a modular and scalable approach.

### Project Overview

The project consists of the following main components:

### Data Module
Implemented using PyTorch Lightning’s LightningDataModule.
Downloads and prepares the MNIST dataset.
Handles splitting into training, validation, and test sets.
Provides PyTorch DataLoaders for batch processing.

### Generator

Neural network that creates fake MNIST images from random noise (latent vectors).
Uses linear and transposed convolution layers to generate 28x28 grayscale images.

### Discriminator
Neural network that distinguishes between real and generated images.
Simple CNN architecture with convolutional layers and fully connected layers.
Outputs a probability of the input being a real image.

### GAN Lightning Module
Combines the Generator and Discriminator.
Implements manual optimization for adversarial training.
Includes adversarial loss calculation for both networks.
Provides image visualization for generated samples after each epoch.

### Training
Training is done with PyTorch Lightning Trainer.
Supports GPU acceleration if available.
Configurable hyperparameters including latent dimension, batch size, and learning rate.
Displays generated images periodically to monitor progress.
### Features
- Modular design using PyTorch Lightning.
- Generator and Discriminator implemented as separate classes.
- Manual optimization for proper GAN training.
- Automatic data handling and splitting.
- Real-time visualization of generated images.
- Easily configurable for CPU or GPU training.

### Requirements.txt
```
torch==2.6.0+cu124
torchvision==0.15.1
pytorch-lightning==2.5.1
torchmetrics==1.7.1
matplotlib==3.8.1
 ``` 
### Installation

1. Clone the repository:

```
git clone <your-repo-url>
cd MNIST_GAN
```
### Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
### Install dependencies:
```
pip install -r requirements.txt
```
### Usage
Train the GAN:
```
python train.py
```
The training script automatically detects GPU if available.
Training runs for 20 epochs by default.
Generate and visualize sample images:
The GAN module includes a plot_imgs() method to visualize generated images.

### Notes
- Dataset (MNIST) is downloaded automatically to data/.
- Logging is handled by PyTorch Lightning.
- The project uses manual optimization for adversarial training.
- The GAN is designed for educational purposes and MNIST-scale experiments.
= If no GPU is available, set accelerator='cpu' in the Lightning Trainer.
= Hyperparameters such as batch size, latent dimension, and learning rate can be adjusted for experimentation.

### References
PyTorch Lightning
MNIST Dataset
GAN concepts: Ian Goodfellow et al., 2014
Install the required dependencies:
```
pip install torch torchvision pytorch-lightning matplotlib
```
#### Optional: Use Google Colab for GPU acceleration.

### Usage
- Initialize the MNIST data module.
- Instantiate the GAN model.
- Train using PyTorch Lightning Trainer.
- Visualize generated images during training to monitor the model's performance.
  
## License
### This repository is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
