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

def load_audio(audiopath, sampling_rate=19000):
    if audiopath is None:
        st.error("Please upload a valid audio file.")
        return None

    try:
        if audiopath.name.endswith('.mp3'):
            audio, lsr = librosa.load(io.BytesIO(audiopath.read()), sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        elif audiopath.name.endswith('.wav'):
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


#classifier function

def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=2, attn_blocks=4, num_attn_heads=4, base_channels=32, dropout=0, kernel_size=5, distribute_zero_label=False)

    state_dict = torch.load('./mel_norms.pth', map_location=torch.device('cpu'))
    # state_dict = torch.load('state_dict.pkl')
    
    # if torch.is_tensor(state_dict):
    #     state_dict = io.BytesIO(state_dict.numpy())

    # classifier.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))

    classifier.load_state_dict(state_dict)

    classifier.eval()
    
    clip = clip.cpu().unsqueeze(0)
    with torch.no_grad():
        results = F.softmax(classifier(clip), dim=1)
    return results[0][0]

st.set_page_config(layout="wide")

# ...

def main():
    
    st.title("Voice Deepfakes Detector")

    upload_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if upload_file is not None:  # Check if a file has been uploaded
        if st.button("Analyze audio"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("Below are the results: ")
                # Load and classify the audio file
                audio_clip = load_audio(upload_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result probability: {result}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI generated.")

            with col2:
                st.audio("Your uploaded audio file is as below:", audio_clip.numpy())  # Pass audio data here
                # Create a waveform
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(
                    title="Waveform plot of your audio file",
                    xaxis_title="Time",
                    yaxis_title="Amplitude",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.info("Disclaimer")
                st.warning("This classification/detection mechanisms are not always accurate. Please do not use this as a sole basis to determine if an audio is AI generated or not. This tool is just to help you get an approximate overview. The results generated should only be considered as a strong signal.")

if __name__ == "__main__":
    main()




