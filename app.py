import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained GAN model
model = tf.keras.models.load_model(
    filepath='/Users/aditya/Documents/FashGanProject /Model/gen.keras',
    custom_objects=None,  # Specify custom objects if needed
    compile=True,  # Whether to compile the model after loading
    safe_mode=True  # Enable safe mode to avoid code execution in custom objects
)
# Function to generate and display an image
def generate_image():
    noise = np.random.normal(0, 1, (1, 128))  # Adjusted noise dimension to 128
    generated_image = model.predict(noise)
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)  # Rescale to [0, 255]
    return generated_image[0]

# Streamlit app
st.title("GAN Image Generator")

if st.button('Generate Image'):
    image = generate_image()
    st.image(image, channels="RGB")
