import streamlit as st
from streamlit_mic_recorder import mic_recorder
from google.cloud import speech

st.title("Voice Input Test")

audio_data = mic_recorder("🔴 Start speaking", "⏹ Stop", format="wav", key="mic")
if audio_data:
    st.audio(audio_data["bytes"])  # optional: play back the audio
    
    # Transcribe with Google Cloud STT
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data["bytes"])
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_data["sample_rate"],
        language_code="en-US"
    )
    result = client.recognize(config=config, audio=audio)
    if result.results:
        text = result.results[0].alternatives[0].transcript
    else:
        text = ""
    st.write("You said:", text)

    # (Then continue with the rest of your app logic using `text`.)
