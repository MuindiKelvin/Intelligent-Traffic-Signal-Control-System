import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import time

def load_model():
    return tf.keras.models.load_model('traffic_model.h5')

def process_image(image, img_height=224, img_width=224):
    img = image.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def get_prediction(model, image):
    class_names = ['accident', 'dense_traffic', 'fire', 'sparse_traffic']
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

def get_signal_timing(traffic_state, confidence):
    base_timings = {
        'sparse_traffic': 30,
        'dense_traffic': 60,
        'accident': 90,
        'fire': 120
    }
    return int(base_timings[traffic_state] * confidence)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_loading_bar():
    progress_bar = st.progress(0)
    status_text = st.empty()
    for percent_complete in range(101):
        progress_bar.progress(percent_complete)
        status_text.text(f"Processing... {percent_complete}%")
        time.sleep(0.03)  
    progress_bar.empty()  
    status_text.empty()  

def main():
    st.markdown("""
    <style>
    .stApp {
        padding: 1rem;
        background-color: #F5F5F5;
    }
    .responsive-image {
        max-width: 100%;
        height: auto;
        border: 3px solid #4CAF50;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-item {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .emergency-alert {
        background-color: #FFEBEE;
        border: 1px solid #FFCDD2;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align: center; color: #333333;">🚦 Traffic Signal Control System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #555555;">Upload an image to analyze traffic conditions</p>', unsafe_allow_html=True)

    model = load_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        display_loading_bar()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            image = Image.open(uploaded_file)
            base64_image = image_to_base64(image)
            st.markdown(f'''
            <div style="text-align: center;">
                <img src="data:image/png;base64,{base64_image}" 
                     alt="Uploaded Image" 
                     class="responsive-image"/>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            processed_image = process_image(image)
            traffic_state, confidence = get_prediction(model, processed_image)
            signal_timing = get_signal_timing(traffic_state, confidence)

            st.markdown(f'''
            <div class="metric-item">
                <h4>Traffic State</h4>
                <p>{traffic_state.replace('_', ' ').title()}</p>
            </div>
            <div class="metric-item">
                <h4>Confidence</h4>
                <p>{confidence:.2%}</p>
            </div>
            <div class="metric-item">
                <h4>Signal Timing</h4>
                <p>{signal_timing} seconds</p>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            if traffic_state in ['accident', 'fire']:
                alert_message = (
                    "🚨 Accident detected! Emergency services should be notified."
                    if traffic_state == 'accident'
                    else "🔥 Fire detected! Emergency protocols activated."
                )
                st.markdown(f'''
                <div class="emergency-alert">
                    <h4>Emergency Alert</h4>
                    <p>{alert_message}</p>
                </div>
                ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
