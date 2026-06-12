import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("train_model.pkl")

st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))

    img_array = np.array(image)

    # Invert if needed
    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")
    st.success(f"Predicted Digit: {digit}")