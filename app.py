import streamlit as st
from PIL import Image

from object_detection import detect_objects
from text_extraction import extract_text
from image_caption import generate_caption
from speech import speak

st.set_page_config(
    page_title="SightMate AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ SightMate AI")
st.subheader("AI-Powered Visual Accessibility Assistant")

st.write(
    "Your personal visual assistant powered by AI."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=500)

    st.divider()

    st.subheader("Choose an AI feature")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting objects..."):
                try:
                    results = detect_objects(image)
                    st.write("Detected Objects")
                    st.json(results)
                except Exception as e:
                    st.error(f"Detection error: {e}")

    with col2:
        if st.button("📝 Read Text"):
            with st.spinner("Extracting text..."):
                try:
                    text = extract_text(image)
                    st.write("Extracted Text")
                    st.write(text)

                    if text.strip():
                        if st.button("🔊 Speak Text"):
                            speak(text)
                except Exception as e:
                    st.error(f"OCR error: {e}")

    with col3:
        if st.button("🖼️ Describe Image"):
            with st.spinner("Generating description..."):
                try:
                    caption = generate_caption(image)
                    st.success(caption)

                    if st.button("🔊 Speak Description"):
                        speak(caption)
                except Exception as e:
                    st.error(f"Caption error: {e}")