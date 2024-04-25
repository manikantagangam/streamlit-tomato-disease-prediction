import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load your trained model
MODEL_PATH = 'Tomato_Disease_Detection.h5'
model = load_model(MODEL_PATH)

def model_predict(img, model):
    # Resize the image to match the model's input shape (224x224)
    img = img.resize((224, 224))
    # Convert the image to a numpy array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    preds = np.argmax(preds, axis=1)

    class_names = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
        "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
        "Healthy"
    ]

    return class_names[int(preds)]

# Streamlit app code
st.title('Tomato Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

disease_treatments = {
    "Bacterial_spot": ["Apply copper-based fungicides", "Prune affected leaves", "Rotate crops"],
    "Early_blight": ["Use fungicides containing mancozeb or copper", "Remove infected plant parts", "Improve air circulation"],
    "Late_blight": ["Apply fungicides containing chlorothalonil", "Prune affected leaves", "Avoid overhead watering"],
    "Leaf_Mold": ["Keep leaves dry", "Prune infected areas", "Avoid overhead watering"],
    "Septoria_leaf_spot": ["Apply fungicides", "Prune affected leaves", "Provide good air circulation"],
    "Spider_mites Two-spotted_spider_mite": ["Use insecticidal soap", "Keep plants well-watered", "Prune affected areas"],
    "Target_Spot": ["Apply fungicides containing copper", "Prune affected leaves", "Remove debris from the garden"],
    "Tomato_Yellow_Leaf_Curl_Virus": ["Use insecticides to control whiteflies", "Remove infected plants", "Use disease-resistant varieties"],
    "Tomato_mosaic_virus": ["Remove and destroy infected plants", "Control aphids", "Avoid working with plants when they are wet"],
    "Healthy": ["No specific treatment needed. Maintain plant health."]
}

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions when a user uploads an image
    prediction = model_predict(image, model)
    # st.write(f"Predicted Disease Class: {prediction}, treatment is {disease_treatments[prediction][0]}")
    st.write(f"Predicted Disease Class: {prediction}")
    st.write("Possible Treatments:")
    for treatment in disease_treatments[prediction]:
        st.write(f"- {treatment}")
