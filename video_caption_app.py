import streamlit as st
import tempfile
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import imageio

# Load pre-trained model and tokenizer (this is just an example, you would need a video captioning model)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def extract_frames(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i % reader.get_meta_data()['fps'] == 0:
            image = Image.fromarray(frame)
            frames.append(image)
    return frames

# Streamlit UI
st.title("Video Captioning App")
st.write("Upload a video and get an AI-generated caption.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    temp_video_path = tfile.name

    st.video(uploaded_file)

    # Extract frames from video
    st.write("Extracting frames from video...")
    frames = extract_frames(temp_video_path)

    # Generate captions for extracted frames
    st.write("Generating captions...")
    captions = []
    for i, frame in enumerate(frames):
        caption = generate_caption(frame)
        captions.append(f"Frame {i + 1}: {caption}")
        st.write(captions[-1])

    # Clean up temporary file
    os.unlink(temp_video_path)

    st.write("Caption generation completed.")
