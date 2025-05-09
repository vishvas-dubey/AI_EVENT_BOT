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
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import io
from pydub import AudioSegment
from st_audiorec import st_audiorec
import os
os.environ["STREAMLIT_WATCHER_NONPYTHON_FILES"] = "false"



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
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%);
    }
    
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border-left: 5px solid #4338CA;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }


    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px -10px rgba(0,0,0,0.15), 0 10px 15px -5px rgba(0,0,0,0.1);
    }
    
    .voice-box {
        background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        margin-bottom: 25px;
        border: 1px solid #C7D2FE;
        position: relative;
        overflow: hidden;
    }
    
    .voice-box::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 70%);
        opacity: 0;
        transition: opacity 0.5s;
        pointer-events: none;
    }
    
    .voice-box:hover::before {
        opacity: 1;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        color: white;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #4338CA, #4F46E5);
        box-shadow: 0 8px 15px rgba(79, 70, 229, 0.4);
        transform: translateY(-3px);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .agenda-item {
        padding: 18px;
        border-radius: 12px;
        background-color: white;
        margin-bottom: 15px;
        border-left: 4px solid #4F46E5;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .agenda-item:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }
    
    .agenda-time {
        font-weight: 600;
        color: #4F46E5;
        font-size: 1.05em;
    }
    
    .highlight {
        background: linear-gradient(90deg, #DBEAFE, #E0E7FF);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 500;
        margin: 0 2px;
    }
    
    /* Chat styling */
    .user-bubble, .bot-bubble {
        padding: 15px 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        max-width: 85%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }
    
    .user-bubble {
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .user-bubble:hover {
        box-shadow: 0 6px 12px rgba(79, 70, 229, 0.25);
    }
    
    .bot-bubble {
        background-color: white;
        color: #1F2937;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        border-left: 4px solid #4F46E5;
    }
    
    .bot-bubble:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .floating-animation {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 12px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4F46E5, #6366F1) !important;
        color: white !important;
        transform: translateY(-5px);
        box-shadow: 0 -4px 15px rgba(79, 70, 229, 0.25) !important;
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    
    /* Custom Input Fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        padding: 12px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div {
        background-color: #4F46E5;
        background-image: linear-gradient(45deg, #4F46E5 25%, #6366F1 25%, #6366F1 50%, #4F46E5 50%, #4F46E5 75%, #6366F1 75%);
        background-size: 20px 20px;
        animation: progress-animation 2s linear infinite;
    }
    
    @keyframes progress-animation {
        0% { background-position: 0 0; }
        100% { background-position: 40px 0; }
    }
    
    /* File Uploader */
    .stFileUploader {
        padding: 15px;
        border-radius: 12px;
        border: 2px dashed #C7D2FE;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #4F46E5;
        background-color: rgba(224, 231, 255, 0.2);
    }
    
    /* Radio Button */
    .stRadio > div {
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Alert Messages */
    .stAlert {
        border-radius: 12px;
        border-left-width: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    
    /* Special decorative elements */
    .decorative-circle {
        position: fixed;
        border-radius: 50%;
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        opacity: 0.1;
        z-index: -1;
    }
    
    .circle-1 {
        width: 300px;
        height: 300px;
        top: -100px;
        right: -100px;
    }
    
    .circle-2 {
        width: 200px;
        height: 200px;
        bottom: -50px;
        left: -50px;
    }

    /* Shimmer effect for buttons */
    .shimmer-button {
        position: relative;
        overflow: hidden;
    }
    
    .shimmer-button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.3) 50%,
            rgba(255, 255, 255, 0) 100%
        );
        transform: rotate(30deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-100%) rotate(30deg);
        }
        100% {
            transform: translateX(100%) rotate(30deg);
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c7d2fe;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a5b4fc;
    }
    
    /* Audio recorder custom styles */
    .css-1n76uvr, .css-18ni7ap {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Chat input box */
    .stChatInput {
        border-radius: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    
    /* Add decorative circles */
    .decorative-circle {
        position: fixed;
        border-radius: 50%;
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        opacity: 0.1;
        z-index: -1;
    }
</style>

<!-- Decorative Circles -->
<div class="decorative-circle circle-1"></div>
<div class="decorative-circle circle-2"></div>
""", unsafe_allow_html=True)

# Create sidebar with enhanced styling
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom: 5px;'>🤖 Event Bot AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em; margin-bottom: 25px;'>Your Personal Hackathon Assistant</p>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; padding: 15px;'><span style='font-size: 90px;' class='floating-animation'>🤖</span></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='background: linear-gradient(to bottom right, #fefefe, #f5f7fa);'>", unsafe_allow_html=True)
    st.markdown("### 🗓️ Event Schedule")
    
    # Add a bit more detail and styling to the schedule
    schedule_items = [
        {"time": "9:00 AM", "event": "Registration", "icon": "📋"},
        {"time": "10:00 AM", "event": "Opening Ceremony", "icon": "🎬"},
        {"time": "1:00 PM", "event": "Lunch Break", "icon": "🍽️"},
        {"time": "6:00 PM", "event": "Demos & Judging", "icon": "🏆"}
    ]
    
    for item in schedule_items:
        st.markdown(f"""
        <div style='padding: 10px; margin-bottom: 10px; border-radius: 8px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <div style='display: flex; align-items: center;'>
                <div style='font-size: 22px; margin-right: 10px;'>{item['icon']}</div>
                <div>
                    <div style='font-weight: bold; color: #4F46E5;'>{item['time']}</div>
                    <div>{item['event']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='background: linear-gradient(to bottom right, #fefefe, #f5f7fa);'>", unsafe_allow_html=True)
    st.markdown("### 📍 Quick Links")
    
    # Enhanced quick links with icons and hover effects
    quick_links = [
        {"name": "Event Map", "url": "https://example.com", "icon": "🗺️"},
        {"name": "Judging Criteria", "url": "https://example.com", "icon": "📊"},
        {"name": "Prizes", "url": "https://example.com", "icon": "🏆"},
        {"name": "Rules", "url": "https://example.com", "icon": "📜"}
    ]
    
    for link in quick_links:
        st.markdown(f"""
        <a href="{link['url']}" target="_blank" style="text-decoration: none; color: inherit;">
            <div style='padding: 10px; margin-bottom: 10px; border-radius: 8px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: all 0.3s ease;' onmouseover="this.style.transform='translateX(5px)'; this.style.boxShadow='0 4px 10px rgba(0,0,0,0.1)';" onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='0 2px 5px rgba(0,0,0,0.05)';">
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: 20px; margin-right: 10px;'>{link['icon']}</div>
                    <div>{link['name']}</div>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add a new countdown timer section
    st.markdown("<div class='card' style='background: linear-gradient(to bottom right, #fefefe, #f5f7fa);'>", unsafe_allow_html=True)
    st.markdown("### ⏱️ Hackathon Countdown")
    
    # Simulate a countdown timer (would need JavaScript for a real one)
    hours_left = 32
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='font-size: 2.5em; font-weight: bold; margin: 10px 0; background: linear-gradient(90deg, #4F46E5, #6366F1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{hours_left}:00:00</div>
        <div style='font-size: 0.9em; opacity: 0.8;'>Hours Remaining</div>
        <div style='width: 100%; height: 8px; background-color: #e5e7eb; border-radius: 4px; margin: 15px 0; overflow: hidden;'>
            <div style='width: 65%; height: 100%; background: linear-gradient(90deg, #4F46E5, #6366F1); border-radius: 4px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content with enhanced styling
st.markdown("<h1 style='text-align: center; font-size: 2.5em; margin-bottom: 30px;'>Welcome to the Hackathon!</h1>", unsafe_allow_html=True)

# Main content
#st.markdown("<h1 style='text-align: center;'>Welcome to the Hackathon!</h1>", unsafe_allow_html=True)

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

# Initialize session state for speech recognition result
if "spoken_name" not in st.session_state:
    st.session_state.spoken_name = ""

# Display authentication section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='voice-box'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Let's verify your registration</h3>", unsafe_allow_html=True)
    st.markdown("<p>Speak or type your full name to get started</p>", unsafe_allow_html=True)
    
    # Voice input
    st.markdown("<div style='text-align: center;'><h4>🎙️ Say your name</h4><div style='font-size: 50px; margin: 20px 0;' class='pulse-animation'>🎤</div></div>", unsafe_allow_html=True)
    
    # Voice recognition using Streamlit Mic Recorder

    audio_data = st_audiorec()

    if audio_data:
        st.audio(audio_data, format="audio/wav")

        # Process and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
            audio.export(temp_audio_file.name, format="wav")

            model = whisper.load_model("small")  # You can change to "small", "medium", etc.
            result = model.transcribe(temp_audio_file.name, language="en")
            print(result["language"])


            # Save to session state
            st.session_state.spoken_name = result["text"]

            st.subheader("📝 Transcription:")
            st.write(result["text"])

    spoken_name = st.session_state.spoken_name
    if spoken_name:
        st.info(f"मैंने सुना: {spoken_name}")
        st.subheader("📝 Transcription:")
        st.write(f"मैंने सुना: **{spoken_name}**")

        confirm = st.radio("क्या यह आपका नाम है?", ("Yes", "No"), key="confirm_radio")
        if confirm == "Yes":
            st.session_state.confirmed_name = spoken_name
        else:
            st.session_state.confirmed_name = ""
            st.warning("कृपया अपना नाम फिर से बोलें या नीचे टाइप करें।")
    else:
        st.info("कृपया अपना नाम बोलें...")

    # Optional manual fallback
    if st.button("मैनुअल माइक्रोफोन टेस्ट"):
        import speech_recognition as sr
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("सुन रहा हूँ...")
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = r.recognize_google(audio, language="hi-IN")
                    st.success(f"आपने कहा: {text}")
                    st.session_state.spoken_name = text
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


# Final user name
user_name = st.session_state.spoken_name.strip() or typed_name.strip()

# Debug information (can be removed in production)
if user_name:
    st.write(f"Detected name: {user_name}")

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