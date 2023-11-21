from distutils.command import upload
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import io
from scipy.io.wavfile import read
from pydub import AudioSegment
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import plotly.express as px

# Set the page title and favicon
st.set_page_config(page_title="Voice Deepfakes Detector", page_icon="🔊")

# Define custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: white;
        height: 100%;
        width: 100%;
    }
    .stApp {
        max-width: 800px auto;
        margin: 0 auto;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4f61ff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2f3cba;
        color: white;
    }
    .stButton>button:active {
        color: white;
    }

    .stbutton>button:focus {
        color: black;
    }
    .stMarkdown {
        line-height: 1.5;
    }
    .stSuccess {
        background-color: #5cb85c;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stInfo {
        background-color: #5bc0de;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #f0ad4e;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #d9534f;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MAX_ALLOWED_DURATION = 6000


def trim_audio(audio, max_duration):
    if len(audio) > max_duration:
        audio = audio[:max_duration]
    return audio


def load_audio(audiopath, sampling_rate=10000):
    if audiopath is None:
        st.error("Please upload a valid audio file.")
        return None

    try:
        if audiopath.name.endswith(".mp3"):
            audio, lsr = librosa.load(io.BytesIO(audiopath.read()), sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        elif audiopath.name.endswith(".wav"):
            audio, lsr = torchaudio.load(audiopath)
            audio = audio[0]
        else:
            st.error(f"Unsupported audio format: {audiopath.name}")
            return None

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        if torch.any(audio > 2) or not torch.any(audio < 0):
            st.warning(
                "Error with audio data. Max={}, min={}".format(audio.max(), audio.min())
            )
        audio.clip_(-1, 1)

        return audio.unsqueeze(0)

    except Exception as e:
        st.error("An error occurred while processing the audio: {}".format(str(e)))
        return None


def classify_audio_clip(clip):
    model1 = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=2,
        attn_blocks=4,
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False,
    )
    model2 = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=2,
        attn_blocks=4,
    )

    # Load the state_dict of both models
    state_dict1 = torch.load("./mel_norms.pth", map_location=torch.device("cpu"))
    state_dict2 = torch.load("./custommodel.pth", map_location=torch.device("cpu"))

    # Load the state_dict into each model
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)

    model1.eval()
    model2.eval()

    clip = clip.cpu().unsqueeze(0)
    with torch.no_grad():
        output1 = model1(clip)
        output2 = model2(clip)

    # Define weights for each model
    model1_weight = 0.7
    model2_weight = 0.3

    # Perform weighted ensembling
    ensembled_output = (output1 * model1_weight) + (output2 * model2_weight)

    # Apply softmax to the ensembled output
    result = F.softmax(ensembled_output, dim=-1)
    return result[0][0]


# Main Streamlit app
st.title("Voice Deepfakes Detector")

st.markdown("Detect AI-generated audio with this simple tool!")

upload_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if upload_file is not None:
    # Load and classify the audio file
    audio_clip = load_audio(upload_file)

    # Check the duration of the audio
    audio_duration = len(audio_clip)
    if audio_duration > MAX_ALLOWED_DURATION:
        st.warning(
            f"Audio duration exceeds the allowed maximum. Trimming to {MAX_ALLOWED_DURATION / 1000} seconds."
        )
        audio_clip = trim_audio(audio_clip, MAX_ALLOWED_DURATION)

    if st.button("🔊 Play audio file"):
        audio_format = "audio/mpeg" if upload_file.name.endswith(".mp3") else "audio/wav"
        audio_np = audio_clip.numpy()
        audio_bytes = audio_np.tobytes()

        # st.write(f"Audio shape: {audio_np.shape}")
        # st.write(f"Audio min value: {audio_np.min()}")
        # st.write(f"Audio max value: {audio_np.max()}")

        st.audio(audio_bytes, format=audio_format, start_time=0)

    if st.button("🔍 Analyze audio"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("🎯 Results:")
            result = classify_audio_clip(audio_clip)
            result = result.item()
            st.info(f"Result probability: {result}")
            st.success(
                f"The uploaded audio is {result * 100:.2f}% likely to be AI generated."
            )

        with col3:
            st.info("📌 Disclaimer")
            st.warning(
                "This classification/detection mechanism is not always accurate. Please do not use this as the sole basis to determine if an audio is AI-generated or not. This tool is just to help you get an approximate overview. The results generated should only be considered as a strong signal."
            )

# Add some space for better visual separation
st.markdown("<hr class='st-ib'>", unsafe_allow_html=True)

st.markdown("Made with ❤️ by your Streamlit expert!")
