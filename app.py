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



# Set page config
st.set_page_config(
    page_title="Event Bot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Enhanced layout for a professional UI

# Add a hero section
st.markdown("""
<div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #333333, #555555); color: white; border-radius: 10px; margin-bottom: 30px;">
    <h1 style="font-size: 3em; margin-bottom: 10px;">Welcome to Event Bot AI</h1>
    <p style="font-size: 1.2em;">Your Personal Hackathon Assistant</p>
    <a href="#" style="text-decoration: none;">
        <button style="background: white; color: #333333; padding: 10px 20px; font-size: 1em; border: none; border-radius: 5px; cursor: pointer;">Get Started</button>
    </a>
</div>
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

    # Feedback form in sidebar
    st.markdown("<div class='card' style='margin-top: 30px; background: linear-gradient(to bottom right, #222, #555); color: #fff;'>", unsafe_allow_html=True)
    st.markdown("### 📝 Event Feedback", unsafe_allow_html=True)
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if not st.session_state.feedback_submitted:
        feedback_questions = [
            "How would you rate the overall event experience?",
            "How helpful were the sessions for your learning?",
            "How was the venue and facilities?",
            "How likely are you to recommend this event to others?",
            "Any suggestions or comments?"
        ]
        feedback_answers = []
        for i, q in enumerate(feedback_questions[:-1]):
            feedback_answers.append(st.radio(q, ["1 - Poor", "2", "3", "4", "5 - Excellent"], key=f"fb_{i}"))
        feedback_answers.append(st.text_area(feedback_questions[-1], key="fb_suggestion"))
        if st.button("Submit Feedback"):
            feedback_data = {f"q{i+1}": ans for i, ans in enumerate(feedback_answers)}
            import json
            import datetime
            feedback_data["timestamp"] = str(datetime.datetime.now())
            with open("feedback_responses.json", "a") as f:
                f.write(json.dumps(feedback_data) + "\n")
            st.success("Thank you for your feedback!")
            st.session_state.feedback_submitted = True
    else:
        st.info("You have already submitted feedback. Thank you!")
    st.markdown("</div>", unsafe_allow_html=True)

    # Developer names section
    st.markdown("""
    <div class='card' style='background: linear-gradient(to bottom right, #23272b, #444); color: #e0e0e0; margin-top: 20px; text-align: center;'>
        <h4 style='margin-bottom: 10px; color: #fff;'>Developed by</h4>
        <ul style='list-style: none; padding: 0; margin: 0; font-size: 1.08em;'>
            <li style='margin-bottom: 6px;'>Vishvas Dubey</li>
            <li style='margin-bottom: 6px;'>Aishwarya Patil</li>
            <li style='margin-bottom: 6px;'>Shuritika Shripat</li>
            <li>Shreeja Sharma</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

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
    
#    confirmed = ["John Doe", "Jane Smith", "Alex Johnson", "Test User"]

# Initialize session state for speech recognition result
if "spoken_name" not in st.session_state:
    st.session_state.spoken_name = ""

# Display authentication section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='voice-box'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Let's verify your registration</h3>", unsafe_allow_html=True)
    st.markdown("<p>Speak or type your full name to get started</p>", unsafe_allow_html=True)

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
        st.session_state.spoken_name = transcript.strip()


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
user_name = st.session_state.spoken_name or typed_name.strip()

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
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Personalized Recommendations", 
            "📋 Event Details", 
            "💬 Ask Me Anything",
            "📝 Event Feedback"
        ])
        
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
        
        # --- LangChain, Gemini, FAISS Setup for RAG ---
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        import tempfile

        # Load and split the document
        loader = TextLoader("document.txt")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # Embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Gemini LLM for QA
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💬 Ask Me Anything About The Event")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            user_input = st.chat_input("Ask about agenda, location, lunch timing, etc. (powered by Gemini+LangChain)")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
                with st.spinner("Thinking..."):
                    response = qa_chain.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📝 Share Your Event Experience")
            
            if "feedback_step" not in st.session_state:
                st.session_state.feedback_step = 0
            if "feedback_answers" not in st.session_state:
                st.session_state.feedback_answers = []
            if "feedback_submitted" not in st.session_state:
                st.session_state.feedback_submitted = False

            if not st.session_state.feedback_submitted:
                # Use predefined questions instead of generating them
                if "feedback_questions" not in st.session_state:
                    st.session_state.feedback_questions = [
                        "What did you enjoy most about today's event?",
                        "Tell me about an interesting conversation you had with a professional today.",
                        "Which technologies or concepts excited you the most?",
                        "How do you plan to apply what you learned today?",
                        "If you could change or improve one thing about the event, what would it be?"
                    ]

                # Show all previous questions and answers in order
                for i in range(st.session_state.feedback_step):
                    st.markdown(f"<div class='bot-bubble'>{st.session_state.feedback_questions[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='user-bubble'>{st.session_state.feedback_answers[i]}</div>", unsafe_allow_html=True)
                
                # Show current question
                current_q = st.session_state.feedback_questions[st.session_state.feedback_step]
                st.markdown(f"<div class='bot-bubble'>{current_q}</div>", unsafe_allow_html=True)

                # Get current answer
                if st.session_state.feedback_step < len(st.session_state.feedback_questions):
                    answer = st.text_area("Your response:", key=f"feedback_{st.session_state.feedback_step}")
                    if st.button("Next", key=f"next_{st.session_state.feedback_step}"):
                        if answer.strip():
                            st.session_state.feedback_answers.append(answer)
                            st.session_state.feedback_step += 1
                            if st.session_state.feedback_step >= len(st.session_state.feedback_questions):
                                # Save feedback
                                feedback_data = {
                                    "timestamp": str(datetime.datetime.now()),
                                    "answers": dict(zip(st.session_state.feedback_questions, st.session_state.feedback_answers))
                                }
                                with open("feedback_responses.json", "a") as f:
                                    f.write(json.dumps(feedback_data) + "\n")
                                st.session_state.feedback_submitted = True
                            st.rerun()
                        else:
                            st.warning("Please provide an answer before continuing.")

            else:
                st.success("Thank you for sharing your valuable feedback! 🙏")
                st.markdown("""
                <div style='text-align: center; margin-top: 20px;'>
                    <div style='font-size: 60px;'>🌟</div>
                    <p>Your responses will help us improve future events!</p>
                </div>
                """, unsafe_allow_html=True)
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

# Add a footer
st.markdown("""
<div class="footer">
    <p>© 2025 Event Bot AI. All rights reserved. | <a href="https://example.com">Privacy Policy</a> | <a href="https://example.com">Terms of Service</a></p>
</div>
""", unsafe_allow_html=True)