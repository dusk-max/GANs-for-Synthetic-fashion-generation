# GANs for Synthetic Fashion Generation
 
## Overview
FashGAN is a Generative Adversarial Network (GAN) designed to generate synthetic fashion images. It uses TensorFlow and TensorFlow Datasets for training on the Fashion MNIST dataset and integrates with Streamlit to provide a web interface for generating images interactively.

---

## Features
- **Generator**: Creates synthetic fashion images using transposed convolutions and upsampling layers.
- **Discriminator**: Classifies images as real or fake using convolutional layers with dropout for better generalization.
- **Training**: Includes noise injection for labels and separate optimizers for generator and discriminator.
- **Interactive Web Interface**: Built using Streamlit to generate images with a single click.

---

## Prerequisites
- Python 3.10 or higher
- TensorFlow 2.x
- Streamlit
- Matplotlib
- TensorFlow Datasets
- Numpy

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/username/FashGAN.git
   cd FashGAN

2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-gpu matplotlib tensorflow-datasets ipywidgets streamlit

3. Train the model:
   Run the GAN training script to generate the models.
   ```bash
   python train_gan.py

4. Launch the Streamlit web app:
   ```bash
   streamlit run app.py

## GAN Architecture

### Generator
- **Input**: Random noise of size 128.
- **Layers**:
  - Dense layer reshaped to 7x7x128.
  - Two upsampling blocks with convolutional layers.
  - Final convolution to produce a 28x28 single-channel image.
- **Activation**: LeakyReLU and sigmoid.

### Discriminator
- **Input**: Image of size 28x28x1.
- **Layers**:
  - Four convolutional blocks with dropout.
  - Dense layer for binary classification.
- **Activation**: LeakyReLU and sigmoid.

## Streamlit Web App

### Steps to Run
1. Save the Streamlit code as `app.py`.
2. Place the trained generator model (`gen.keras`) in the specified path.
3. Run the app:  
   ```bash
   streamlit run app.py

## Results
- The generator model successfully creates synthetic fashion images resembling items from the Fashion MNIST dataset.
- The Streamlit app allows users to generate and view images interactively.

## Training

### 1. Dataset Preprocessing:
- Images scaled to [0, 1].
- Batched and shuffled for training.

### 2. Loss Functions:
- Binary cross-entropy for both generator and discriminator.

### 3. Optimizers:
- Adam optimizers with different learning rates for generator and discriminator.

### 4. Callbacks:
- Custom callback `ModelMonitor` to save generated images at each epoch.

### 5. Training Command:
Run the following command to start training:
```python
fashgan.fit(ds, epochs=20, callbacks=[ModelMonitor()]) 
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contact

For questions or suggestions, please contact at [adityaworks18@gmail.com](mailto:adityaworks18@gmail.com).
