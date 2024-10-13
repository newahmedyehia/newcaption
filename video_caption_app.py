import streamlit as st
import tempfile
import os
import openai_whisper as whisper
import srt
import datetime

# Load Whisper model
model = whisper.load_model("base")

# Streamlit UI
st.title("Video to Subtitle Generator")
st.write("Upload a video and get an SRT file with subtitles.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    temp_video_path = tfile.name

    st.video(uploaded_file)

    # Transcribe audio from video using Whisper
    st.write("Transcribing audio from video...")
    result = model.transcribe(temp_video_path)

    # Generate SRT file from transcription
    st.write("Generating SRT file...")
    segments = result["segments"]
    subtitles = []
    for segment in segments:
        start = datetime.timedelta(seconds=segment["start"])
        end = datetime.timedelta(seconds=segment["end"])
        content = segment["text"]
        subtitle = srt.Subtitle(index=len(subtitles) + 1, start=start, end=end, content=content)
        subtitles.append(subtitle)

    srt_content = srt.compose(subtitles)

    # Save SRT file to a temporary file
    srt_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".srt").name
    with open(srt_file_path, "w") as srt_file:
        srt_file.write(srt_content)

    # Provide SRT file for download
    st.write("Transcription completed. Download your SRT file below.")
    with open(srt_file_path, "rb") as file:
        st.download_button(label="Download SRT File", data=file, file_name="subtitles.srt", mime="text/plain")

    # Clean up temporary files
    os.unlink(temp_video_path)
    os.unlink(srt_file_path)
