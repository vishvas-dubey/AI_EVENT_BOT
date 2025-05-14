import streamlit as st
import json
import os
from datetime import datetime
import re
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# Set page config
st.set_page_config(page_title="Event Bot AI", layout="centered")
st.title("🤖 Event Bot AI – Your Hackathon Assistant")

# Custom CSS for better styling
st.markdown("""
<style>
.voice-box {
    background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.mic-icon {
    font-size: 40px;
    color: #4CAF50;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.7; }
    100% { transform: scale(1); opacity: 1; }
}
.start-button {
    background-color: #ff5252;
    color: white;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
    border-radius: 8px;
    margin-top: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>

<div class="voice-box">
    <div class="mic-icon">🎤</div>
    <h4>Speak or type your name to proceed</h4>
    <p>Click the start button and say your name</p>
    <button class="start-button">🎙️ Start Listening</button><br>
    <small style="color: gray;">or select device from below</small>
</div>
""", unsafe_allow_html=True)

# Load agenda, location, and confirmed user list
with open("agenda.json", "r") as f:
    agenda = json.load(f)

with open("location.json", "r") as f:
    locations = json.load(f)

with open("confirmed_users.json", "r") as f:
    confirmed = json.load(f)["confirmed_users"]

# Capture voice input using streamlit-webrtc
st.subheader("🎙️ Speak or type your name to proceed")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.result_text = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            audio_data = frame.to_ndarray()
            with sr.AudioFile(sr.AudioData(audio_data.tobytes(), frame.sample_rate, 2)) as source:
                audio = self.recognizer.record(source)
                self.result_text = self.recognizer.recognize_google(audio)
        except:
            self.result_text = ""
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

spoken_name = ""
if webrtc_ctx.audio_processor:
    spoken_name = webrtc_ctx.audio_processor.result_text

# Show both voice result and text input
typed_name = st.text_input("Or type your full name below:")

# Use whichever is non-empty
user_name = spoken_name.strip() or typed_name.strip()

# Confirm registration
if user_name:
    if user_name in confirmed:
        st.success(f"✅ Welcome {user_name}! You are confirmed for the event 🎉")

        # Resume parser
        def extract_keywords(text):
            return re.findall(r"\b(?:AI|ML|Python|Data|Cloud|LLM|Vision|NLP|RAG|Vertex)\b", text, flags=re.IGNORECASE)

        def match_sessions(keywords):
            matched = []
            for item in agenda["Day 1"]:
                if any(kw.lower() in item["topic"].lower() for kw in keywords) and item not in matched:
                    matched.append(item)
            return matched

        st.subheader("📄 Upload Your Resume (PDF/Text)")
        resume = st.file_uploader("Upload resume to get session recommendations", type=["txt", "pdf"])
        if resume:
            raw_text = resume.read().decode(errors="ignore")
            skills = extract_keywords(raw_text)
            matches = match_sessions(skills)
            st.success(f"✅ Skills detected: {', '.join(set(skills))}")
            if matches:
                st.info("🎯 Recommended Sessions for You:")
                for item in matches:
                    st.markdown(f"- **{item['time']}** – {item['topic']}")
            else:
                st.warning("No matching sessions found. Try using a more detailed resume.")

        st.subheader("💬 Ask Me Anything")
        user_input = st.chat_input("Ask about agenda, location, lunch timing...")
        if user_input:
            st.chat_message("user").write(user_input)
            response = ""

            if "agenda" in user_input.lower():
                response = "**Today's Agenda:**\n"
                for item in agenda["Day 1"]:
                    response += f"- {item['time']}: {item['topic']}\n"
            elif "washroom" in user_input.lower():
                response = f"The washroom is located: {locations['washroom']}"
            elif "lunch" in user_input.lower():
                response = f"Lunch is served: {locations['lunch']}"
            elif "helpdesk" in user_input.lower():
                response = f"The helpdesk is located: {locations['helpdesk']}"
            elif "time left" in user_input.lower():
                now = datetime.now()
                lunch_time = datetime(now.year, now.month, now.day, 13, 0, 0)
                diff = lunch_time - now
                if diff.total_seconds() > 0:
                    minutes = int(diff.total_seconds() // 60)
                    response = f"⏱️ {minutes} minutes left until lunch!"
                else:
                    response = "🍽️ Lunch time has already started!"
            else:
                response = "Sorry, I can help with agenda, washroom, lunch time, and resume suggestions. Try again!"

            st.chat_message("assistant").write(response)
    else:
        st.error("❌ Sorry, your name is not in the confirmed attendees list.")
