import numpy as np
import os
from pydub import AudioSegment

import streamlit as st
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import plotly.express as px
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torch
from scipy.io.wavfile import read

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


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
            st.warning(f"Error with audio data. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)

        return audio.unsqueeze(0)

    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
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
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False,
    )
# Your SimpleRNN model

    # Load the state_dict of both models
    state_dict1 = torch.load("./mel_norms.pth", map_location=torch.device("cpu"))
    state_dict2 = torch.load("./model_new.pth", map_location=torch.device("cpu"))

    # Load the state_dict into each model
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)

    model1.eval()
    model2.eval()

    clip = clip.cpu().unsqueeze(0)

    clip = clip.permute(0, 2, 1)

    clip = clip.view(clip.size(0), 1, -1)
    
    with torch.no_grad():
        output1 = model1(clip)
        output2 = model2(clip)

    # Define weights for each model
    model1_weight = 0.6
    model2_weight = 0.4

    # Perform weighted ensembling
    ensembled_output = (output1 * model1_weight) + ((output2 * model2_weight)/2)

    # Apply softmax to the ensembled output
    result = F.softmax(ensembled_output, dim=-1)
    return result[0][0]


st.set_page_config(layout="wide")

def main():
    st.title("Voice Deepfakes Detector")

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

        if st.button("Play audio file"):
            audio_numpy = audio_clip.squeeze().numpy()
            st.audio(audio_numpy, format="audio/wav", sample_rate=10000)

        if st.button("Analyze audio"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("Below are the results: ")
                result = classify_audio_clip(audio_clip)
                result = result.item()
                # st.info(f"Result probability: {result}")
                st.success(
                    f"The uploaded audio is {result * 100:.2f}% likely to be AI generated."
                )

            with col3:
                st.info("Disclaimer")
                st.warning(
                    "This classification/detection mechanism is not always accurate. Please do not use this as the sole basis to determine if an audio is AI-generated or not. This tool is just to help you get an approximate overview. The results generated should only be considered as a strong signal."
                )


if __name__ == "__main__":
    main()