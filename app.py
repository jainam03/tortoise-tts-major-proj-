# import streamlit as st
# from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
# import io
# import librosa
# import plotly.express as px
# import torch.nn.functional as F
# import torchaudio
# import torch
# from scipy.io.wavfile import read

# def load_audio(audiopath, sampling_rate=19000):
#     if isinstance(audiopath, str):
#         if audiopath.endswith('.mp3'):
#             audio, lsr = librosa.load(audiopath, sr=sampling_rate)
#             audio = torch.FloatTensor(audio)
#         else:
#             assert False, f"Unsupported audio format: {audiopath}"
#     elif isinstance(audiopath, io.BytesIO):
#         audio, lsr = torchaudio.load(audiopath)
#         audio = audio[0]

#     if lsr != sampling_rate:
#         audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

#     if torch.any(audio > 2) or not torch.any(audio < 0):
#         print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
#     audio.clip_(-1, 1)

#     return audio.unsqueeze(0)


# #classifier function

# def classify_audio_clip(clip):
#     classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=2, attn_blocks=4, num_attn_heads=4, base_channels=32, dropout=0, kernel_size=5, distribute_zero_label=False)

#     state_dict = torch.load('./mel_norms.pth', map_location=torch.device('cpu'))
#     # state_dict = torch.load('state_dict.pkl')
    
#     # if torch.is_tensor(state_dict):
#     #     state_dict = io.BytesIO(state_dict.numpy())

#     # classifier.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))

#     classifier.load_state_dict(state_dict)

#     classifier.eval()
    
#     clip = clip.cpu().unsqueeze(0)
#     with torch.no_grad():
#         results = F.softmax(classifier(clip), dim=1)
#     return results[0][0]

# st.set_page_config(layout="wide")

# def main():
    
#     st.title("Voice Deepfakes Detector")

#     upload_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

#     if upload_file is not None:
#         if st.button("Analyze audio"):
#             col1, col2, col3 = st.columns(3)

#             with col1:
#                 st.info("Below are the results: ")
#                 #load and classify the audio file
#                 audio_clip = load_audio(upload_file)
#                 result = classify_audio_clip(audio_clip)
#                 result = result.item()
#                 st.info(f"Result probability: {result}")
#                 st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI generated.")

#             with col2:
#                 st.audio("Your uploaded audio file is as below: ")
#                 st.audio(upload_file)
#                 #create a waveform
#                 fig = px.line()
#                 fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
#                 fig.update_layout(
#                     title = "Waveform plot of your audio file",
#                     xaxis_title = "Time",
#                     yaxis_title = "Amplitude",
#                     font=dict(
#                         family="Courier New, monospace",
#                         size=18,
#                         color="#7f7f7f"
#                     )
#                 )

#                 st.plotly_chart(fig, use_container_width=True)

#             with col3:
#                 st.info("Disclaimer")
#                 st.warning("This classification/detection mechanisms are not always accurate. Please do not use this as a sole basis to determine if an audio is AI generated or not. This tool is just to help you get an approximate overview. The results generated should only be considered as a strong signal.")


# if __name__ == "__main__":
#     main()


import streamlit as st
import torchaudio
import torch.nn.functional as F
import torch
from scipy.io.wavfile import read
from app_optimized import load_models
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import plotly.express as px
import numpy as np

def classify_audio_stream(stream, model1, model2):
  """Classifies an audio stream using a pre-trained model.

  Args:
    stream: A torchaudio.io.StreamDataset object.
    model1: A pre-trained model for classifying audio.
    model2: A pre-trained model for classifying audio.

  Returns:
    A float representing the probability that the audio is AI generated.
  """

  model1.eval()
  model2.eval()

  # Iterate over the audio stream in chunks.
  results = []
  for chunk in stream:
    # Preprocess the audio chunk.
    chunk = chunk.cpu().unsqueeze(0)

    # Classify the audio chunk.
    with torch.no_grad():
      result1 = F.softmax(model1(chunk), dim=1)
      result2 = F.softmax(model2(chunk), dim=1)

    # Add the results to the list of results.
    results.append(result1.item())
    results.append(result2.item())

  # Calculate the final result.
  result = np.mean(results)
  return result

def main():
  st.title("Voice Deepfakes Detector")

  upload_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

  if upload_file is not None:  # Check if a file has been uploaded
    if st.button("Analyze audio"):
      # Load the audio file as a stream.
      stream = torchaudio.io.StreamDataset(
          upload_file,
          sample_rate=16000,
          buffer_size=1024,
          num_buffers=10,
      )

      # Classify the audio stream.
      model1, model2 = load_models()
      result = classify_audio_stream(stream, model1, model2)

      # Show the results.
      st.info(f"Result probability: {result:.2f}")
      st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI generated.")

