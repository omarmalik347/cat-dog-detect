import streamlit as st
import openai
from PIL import Image
import io

# Set up OpenAI API Key
openai.api_key = "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"  # Replace with your actual API key

# Function to classify the image using GPT-4 Vision API
def classify_with_gpt(image):
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Call OpenAI Vision API
    response = openai.client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert image classifier that identifies whether an image contains a cat or a dog."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a cat or a dog? Answer with just 'Cat' or 'Dog'."},
                    {"type": "image", "image": image_bytes}
                ]
            }
        ],
        max_tokens=5  # Short response
    )

    # Extract response
    return response.choices[0].message.content

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier (GPT-4 Vision)")
st.write("Upload an image, and GPT-4 Vision will classify it as a Cat or Dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify image using GPT-4 Vision
    st.write("🔍 **Classifying...**")
    with st.spinner("Processing with GPT..."):
        label = classify_with_gpt(image)

    # Display classification result
    st.write(f"✅ **The image is classified as:** {label}")
