import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import io
from pydub import AudioSegment

st.title("🎙️ Voice to Text with Whisper (No API Needed)")

audio_data = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop", format="wav", key="recorder")

if audio_data:
    st.audio(audio_data["bytes"], format="audio/wav")

    # Save the audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        # Convert to a format Whisper understands
        audio = AudioSegment.from_file(io.BytesIO(audio_data["bytes"]), format="wav")
        audio.export(temp_audio_file.name, format="wav")

        # Load Whisper model
        model = whisper.load_model("base")  # options: tiny, base, small, medium, large
        result = model.transcribe(temp_audio_file.name)

        st.subheader("📝 Transcription:")
        st.write(result["text"])
        print("🖨️ Transcription (printed in terminal/log):", result["text"])  # 👈 Add this line
