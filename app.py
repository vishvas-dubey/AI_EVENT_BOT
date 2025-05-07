import streamlit as st
import json
import os
from datetime import datetime
import re
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import time
import queue
from typing import List, Optional

# Set page config
st.set_page_config(
    page_title="Event Bot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    h1, h2, h3 {
        color: #1E3A8A;
        font-weight: 600;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%);
    }
    
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #4338CA;
    }
    
    .voice-box {
        background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #C7D2FE;
    }
    
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        border-radius: 8px;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.35);
        transform: translateY(-2px);
    }
    
    .agenda-item {
        padding: 15px;
        border-radius: 8px;
        background-color: white;
        margin-bottom: 10px;
        border-left: 4px solid #4F46E5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    
    .agenda-time {
        font-weight: 600;
        color: #4F46E5;
    }
    
    .highlight {
        background-color: #DBEAFE;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Chat styling */
    .user-bubble, .bot-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .user-bubble {
        background-color: #4F46E5;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .bot-bubble {
        background-color: white;
        color: #1F2937;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        border-left: 3px solid #4F46E5;
    }
    
    .pulse-animation {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🤖 Event Bot AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your Personal Hackathon Assistant</p>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; padding: 20px;'><span style='font-size: 80px;' class='pulse-animation'>🤖</span></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗓️ Event Schedule")
    st.markdown("- 9:00 AM - Registration")
    st.markdown("- 10:00 AM - Opening Ceremony")
    st.markdown("- 1:00 PM - Lunch Break")
    st.markdown("- 6:00 PM - Demos & Judging")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📍 Quick Links")
    st.markdown("- [Event Map](https://example.com)")
    st.markdown("- [Judging Criteria](https://example.com)")
    st.markdown("- [Prizes](https://example.com)")
    st.markdown("- [Rules](https://example.com)")
    st.markdown("</div>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align: center;'>Welcome to the Hackathon!</h1>", unsafe_allow_html=True)

# Load data
try:
    with open("agenda.json", "r") as f:
        agenda = json.load(f)

    with open("location.json", "r") as f:
        locations = json.load(f)

    with open("confirmed_users.json", "r") as f:
        confirmed = json.load(f)["confirmed_users"]
except Exception as e:
    # Create sample data if files don't exist
    agenda = {
        "Day 1": [
            {"time": "09:00 - 10:00", "topic": "Registration & Breakfast"},
            {"time": "10:00 - 11:00", "topic": "Kickoff & Introduction to AI/ML"},
            {"time": "11:00 - 12:00", "topic": "Python for Data Science Workshop"},
            {"time": "12:00 - 13:00", "topic": "Cloud Computing with AI"},
            {"time": "13:00 - 14:00", "topic": "Lunch Break"},
            {"time": "14:00 - 15:30", "topic": "Building LLM Applications"},
            {"time": "15:30 - 17:00", "topic": "Computer Vision & NLP Workshop"},
            {"time": "17:00 - 18:00", "topic": "RAG Systems with Vertex AI"}
        ]
    }
    
    locations = {
        "washroom": "Ground Floor, Near Elevator",
        "lunch": "2nd Floor, Cafeteria",
        "helpdesk": "Main Entrance, Registration Desk"
    }
    
    confirmed = ["John Doe", "Jane Smith", "Alex Johnson", "Test User"]

# Display authentication section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='voice-box'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Let's verify your registration</h3>", unsafe_allow_html=True)
    st.markdown("<p>Speak or type your full name to get started</p>", unsafe_allow_html=True)
    
    # Voice input
    st.markdown("<div style='text-align: center;'><h4>🎙️ Say your name</h4><div style='font-size: 50px; margin: 20px 0;' class='pulse-animation'>🎤</div></div>", unsafe_allow_html=True)
    
    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.recognizer = sr.Recognizer()
            self.result_text = ""
            
            # बेहतर वॉइस डिटेक्शन के लिए पैरामीटर्स
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5
            
            # ऑडियो फ्रेम्स के लिए क्यू
            self._frames = []
            self._audio_queue = queue.Queue()
            self._sample_rate = None
            
            # प्रोसेसिंग के लिए ज़रूरी पैरामीटर्स
            self._num_channels = None
            self._bytes_per_sample = None
            
        def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
            if len(frames) == 0:
                return []
                
            # Get audio parameters from first frame
            frame = frames[0]
            if self._sample_rate is None:
                self._sample_rate = frame.sample_rate
            if self._num_channels is None:
                self._num_channels = frame.layout.nb_channels
            if self._bytes_per_sample is None:
                self._bytes_per_sample = 2  # 16-bit audio
                
            # Collect audio frames
            for frame in frames:
                self._frames.append(frame.to_ndarray().flatten().tobytes())
                
            # Collect enough audio data before processing (1 second)
            buffer_duration = 1.0  # 1 second
            expected_buffer_size = int(self._sample_rate * buffer_duration)
            
            # Process audio if we have enough data
            current_buffer_size = sum(len(f) // self._bytes_per_sample for f in self._frames)
            
            if current_buffer_size >= expected_buffer_size:
                self._process_audio()
                
            return frames
        
        def _process_audio(self):
            try:
                # Combine all audio frames
                audio_data = b''.join(self._frames)
                
                # Convert to AudioData
                audio = sr.AudioData(
                    audio_data,
                    sample_rate=self._sample_rate,
                    sample_width=self._bytes_per_sample
                )
                
                # Reset frames buffer after processing
                self._frames = []
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio, language="hi-IN")
                
                if text:
                    print(f"Recognized: {text}")
                    self.result_text = text
                else:
                    print("No text recognized.")
                    
            except sr.UnknownValueError:
                print("Could not understand audio - speech unintelligible")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(f"Error processing audio: {e}")


    # मुख्य कोड में इस तरह उपयोग करें
    webrtc_ctx = webrtc_streamer(
        key="speech",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    spoken_name = ""
    if webrtc_ctx.audio_processor:
        spoken_name = webrtc_ctx.audio_processor.result_text
        if spoken_name:
            st.info(f"मैंने सुना: {spoken_name}")
        else:
            st.info("कृपया अपना नाम बोलें... (आवाज़ का इनपुट लिया जा रहा है)")
            
    # वैकल्पिक: मैनुअल माइक्रोफोन बटन
    if st.button("मैनुअल माइक्रोफोन टेस्ट"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("सुन रहा हूँ...")
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = r.recognize_google(audio, language="hi-IN")
                    st.success(f"आपने कहा: {text}")
                    # ऑटोमैटिक रूप से इनपुट फिल्ड में डालें
                    if text:
                        spoken_name = text
                except sr.UnknownValueError:
                    st.error("माफ़ करें, मैं आपकी आवाज़ नहीं समझ पाया")
                except sr.RequestError:
                    st.error("Google Speech API से कनेक्शन में समस्या है")
        except Exception as e:
            st.error(f"माइक्रोफोन एक्सेस में समस्या: {e}")
    
    st.markdown("<h4>OR</h4>", unsafe_allow_html=True)
    typed_name = st.text_input("Type your full name:", placeholder="e.g. John Doe")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div style="font-size: 100px; text-align: center;" class="pulse-animation">👋</div>
        </div>
        <div style="text-align: center; margin-top: 20px;">
            <h3>Welcome!</h3>
        </div>
    """, unsafe_allow_html=True)

# Use whichever is non-empty
user_name = spoken_name.strip() or typed_name.strip()

# Confirm registration
if user_name:
    if user_name.lower() in [name.lower() for name in confirmed]:
        # Find the actual case-sensitive name from the list
        for name in confirmed:
            if name.lower() == user_name.lower():
                user_name = name
                break
                
        # st.balloons()
        time.sleep(0.5)
        st.success(f"✅ Welcome {user_name}! You are confirmed for the event 🎉")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["🎯 Personalized Recommendations", "📋 Event Details", "💬 Ask Me Anything"])
        
        with tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📄 Upload Your Resume for Personalized Recommendations")
            st.write("Let me analyze your resume to suggest the most relevant sessions!")
            
            # Resume parser
            def extract_keywords(text):
                return re.findall(r"\b(?:AI|ML|Python|Data|Cloud|LLM|Vision|NLP|RAG|Vertex)\b", text, flags=re.IGNORECASE)

            def match_sessions(keywords):
                matched = []
                for item in agenda["Day 1"]:
                    if any(kw.lower() in item["topic"].lower() for kw in keywords) and item not in matched:
                        matched.append(item)
                return matched

            resume = st.file_uploader("Upload resume (PDF/Text)", type=["txt", "pdf"])
            if resume:
                with st.spinner("Analyzing your skills..."):
                    time.sleep(1.5)  # Simulate processing
                    raw_text = resume.read().decode(errors="ignore")
                    skills = extract_keywords(raw_text)
                    matches = match_sessions(skills)
                    
                    if skills:
                        st.markdown("#### 🔍 Skills Detected:")
                        skills_html = ", ".join([f"<span class='highlight'>{skill}</span>" for skill in set(skills)])
                        st.markdown(f"<p>{skills_html}</p>", unsafe_allow_html=True)
                        
                        if matches:
                            st.markdown("#### 🎯 Recommended Sessions:")
                            for item in matches:
                                st.markdown(f"""
                                <div class='agenda-item'>
                                    <span class='agenda-time'>{item['time']}</span><br>
                                    {item['topic']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No specific matches found. Consider attending our intro sessions!")
                    else:
                        st.warning("No tech skills detected. Try a more detailed resume or check all sessions below.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📅 Today's Agenda")
                for item in agenda["Day 1"]:
                    st.markdown(f"""
                    <div class='agenda-item'>
                        <span class='agenda-time'>{item['time']}</span><br>
                        {item['topic']}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📍 Important Locations")
                
                for key, value in locations.items():
                    icon = "🚻" if key == "washroom" else "🍽️" if key == "lunch" else "❓"
                    st.markdown(f"**{icon} {key.capitalize()}**: {value}")
                
                # Time until lunch
                now = datetime.now()
                lunch_time = datetime(now.year, now.month, now.day, 13, 0, 0)
                diff = lunch_time - now
                
                st.markdown("### ⏱️ Time Until Lunch")
                if diff.total_seconds() > 0:
                    minutes = int(diff.total_seconds() // 60)
                    hours = minutes // 60
                    mins = minutes % 60
                    if hours > 0:
                        st.markdown(f"**{hours}h {mins}m** remaining until lunch!")
                    else:
                        st.markdown(f"**{mins} minutes** remaining until lunch!")
                    
                    # Progress bar
                    morning_mins = 4 * 60  # 9am to 1pm = 4 hours
                    elapsed = morning_mins - minutes
                    progress = elapsed / morning_mins
                    progress = max(0.0, min(1.0, progress))
                    st.progress(progress)
                else:
                    st.success("🍽️ Lunch time has already started!")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💬 Ask Me Anything About The Event")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            
            # Chat input
            user_input = st.chat_input("Ask about agenda, location, lunch timing...")
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
                
                # Generate response
                with st.spinner("Thinking..."):
                    time.sleep(0.5)  # Simulate thinking
                    response = ""
                    
                    if any(word in user_input.lower() for word in ["agenda", "schedule", "program", "sessions"]):
                        response = "<strong>📅 Today's Agenda:</strong><br>"
                        for item in agenda["Day 1"]:
                            response += f"• <span class='agenda-time'>{item['time']}</span>: {item['topic']}<br>"
                    
                    elif any(word in user_input.lower() for word in ["washroom", "toilet", "bathroom", "restroom"]):
                        response = f"🚻 The washroom is located at: <strong>{locations['washroom']}</strong>"
                    
                    elif any(word in user_input.lower() for word in ["lunch", "food", "eat", "meal"]):
                        response = f"🍽️ Lunch is served at: <strong>{locations['lunch']}</strong> from 1:00 PM to 2:00 PM"
                    
                    elif any(word in user_input.lower() for word in ["help", "helpdesk", "assistance", "support", "question"]):
                        response = f"❓ The helpdesk is located at: <strong>{locations['helpdesk']}</strong>"
                    
                    elif any(word in user_input.lower() for word in ["time", "lunch time", "time left", "remaining"]):
                        now = datetime.now()
                        lunch_time = datetime(now.year, now.month, now.day, 13, 0, 0)
                        diff = lunch_time - now
                        if diff.total_seconds() > 0:
                            minutes = int(diff.total_seconds() // 60)
                            hours = minutes // 60
                            mins = minutes % 60
                            if hours > 0:
                                response = f"⏱️ <strong>{hours}h {mins}m</strong> remaining until lunch!"
                            else:
                                response = f"⏱️ <strong>{mins} minutes</strong> remaining until lunch!"
                        else:
                            response = "🍽️ Lunch time has already started!"
                    
                    elif any(word in user_input.lower() for word in ["wifi", "internet", "connection"]):
                        response = "🌐 <strong>WiFi Details:</strong><br>Network: HackathonEvent<br>Password: Hack2025!"
                    
                    elif any(word in user_input.lower() for word in ["prizes", "reward", "win", "awards"]):
                        response = "🏆 <strong>Prize Information:</strong><br>• 1st Place: $5000<br>• 2nd Place: $2500<br>• 3rd Place: $1000<br>• Best UI/UX: $500"
                        
                    else:
                        response = "I can help with information about the agenda, washrooms, lunch, helpdesk, WiFi, prizes, and time until next events. How else can I assist you?"
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("❌ Sorry, your name is not in the confirmed attendees list. Please check with the registration desk.")
        
        # Add a way to see the list of confirmed attendees (FOR DEMO PURPOSES ONLY)
        if st.button("Show confirmed attendees (Demo only)"):
            st.write("Confirmed attendees:", ", ".join(confirmed))
            st.info("This is only shown for demonstration purposes. In a real event, this button wouldn't exist.")
            
        # Provide a fallback option
        st.markdown("""
        <div class='card'>
        <h3>Can't find your name?</h3>
        <p>If you've registered but your name isn't showing up, please visit the registration desk with your confirmation email.</p>
        </div>
        """, unsafe_allow_html=True)