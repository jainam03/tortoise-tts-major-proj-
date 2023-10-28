import numpy as np
import streamlit as st
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import plotly.express as px
import torch.nn.functional as F
import torchaudio
import torch
from scipy.io.wavfile import read


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
            st.warning(f"Error with audio data. Max={audio.max()} min={audio.min()}")
            audio.clip_(-1, 1)

        return audio.unsqueeze(0)

    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
        return None


# classifier function


def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(
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

    state_dict = torch.load("./mel_norms.pth", map_location=torch.device("cpu"))
    # state_dict = torch.load('state_dict.pkl')

    # if torch.is_tensor(state_dict):
    #     state_dict = io.BytesIO(state_dict.numpy())

    # classifier.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))

    classifier.load_state_dict(state_dict)

    classifier.eval()

    clip = clip.cpu().unsqueeze(0)
    with torch.no_grad():
        results = F.softmax(classifier(clip), dim=1)
    return results[0][0]


st.set_page_config(layout="wide")

# Sidebar

sidebar = st.sidebar

# List of uploaded audio clips
uploaded_audio_clips = []

# Add a button to upload an audio file
sidebar.button(
    "Upload audio file",
    on_click=lambda: uploaded_audio_clips.append(
        sidebar.file_uploader("Upload audio file", type=["wav", "mp3"])
    ),
)

# Main content

# Get the first uploaded audio clip, or None if there are no uploaded audio clips
selected_audio_clip = uploaded_audio_clips[0] if uploaded_audio_clips else None

# If there is a selected audio clip, calculate the spectrogram and display it on the page
if selected_audio_clip:
    spectrogram = librosa.core.power_to_db(
        librosa.core.stft(selected_audio_clip.squeeze())
    )

    # Create a Plotly Express figure to display the spectrogram
    fig = px.imshow(spectrogram, origin="lower")
    fig.update_layout(
        title="Spectrogram of selected audio clip",
        xaxis_title="Time",
        yaxis_title="Frequency",
        aspect="equal",
    )

    # Display the spectrogram
    st.plotly_chart(fig, use_container_width=True)

# Otherwise, display a message informing the user that there are no uploaded audio clips
else:
    st.info("Please upload an audio file to see the spectrogram.")
