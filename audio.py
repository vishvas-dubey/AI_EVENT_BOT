import streamlit as st
from streamlit_mic_recorder import mic_recorder
from google.cloud import speech
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Set Google Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
speech_client = speech.SpeechClient()

st.title("🎙️ AI Event Bot: Voice → Transcript → Gemini Insight")

# Mic Recorder
audio_data = mic_recorder(start_prompt="🎤 Speak", stop_prompt="🛑 Stop", format="wav", key="mic")

if audio_data:
    st.success("✅ Audio captured")

    audio_bytes = audio_data["bytes"]
    sr = audio_data["sample_rate"]

    # --- Google Speech-to-Text ---
    g_audio = speech.RecognitionAudio(content=audio_bytes)
    g_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        language_code="en-US"
    )

    st.subheader("🗣️ Google STT Transcript:")
    g_response = speech_client.recognize(config=g_config, audio=g_audio)
    print("Google STT response:", g_response)

    if g_response.results:
        transcript = g_response.results[0].alternatives[0].transcript
        st.write(transcript)
        print("Transcript:", transcript)
        st.success("✅ Transcript generated")

        # Gemini: Summarize or improve
        try:
            prompt = f"Please improve or rephrase this transcript: '{transcript}'"
            gemini_response = gemini_model.generate_content(prompt)
            st.subheader("🔮 Gemini Insight:")
            print("Gemini response:", gemini_response)
            st.write(gemini_response.text)
        except Exception as e:
            st.error(f"Gemini error: {e}")
    else:
        st.warning("No transcript from Google STT.")
