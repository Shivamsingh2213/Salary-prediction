# pip install streamlit
# pip install transformers
# pip install Pillow

import streamlit as st
from transformers import pipeline
from PIL import Image

# Create the caption pipeline
caption = pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en")

# Display the image using Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate the caption
    if st.button("Generate Caption"):
        captions = caption(image)
        st.write(captions[0]["generated_text"])