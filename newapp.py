# --- Imports ---
import streamlit as st
import json
import os
from datetime import datetime
import re
from dotenv import load_dotenv
import time  # Add this import
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from google.cloud import speech
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Move these helper functions to the top of the file, after imports
def extract_candidate_info(text):
    """Extract detailed candidate information"""
    info = {
        "name": "",
        "experience": 0,
        "skills": [],
        "projects": [],
        "company": "",
        "email": "",
        "contact": ""
    }
    
    # Extract name
    name_match = re.search(r'name\s*[-:]\s*([^\n]+)', text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).strip()
    
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        info["email"] = email_match.group(0)
    
    # Extract company
    company_match = re.search(r'company\s*[-:]\s*([^\n]+)', text, re.IGNORECASE)
    if company_match:
        info["company"] = company_match.group(1).strip()
    
    # Extract other fields (experience, skills, projects)
    # ... (existing extraction code) ...
    
    return info

def load_resumes(resume_folder="resumes"):
    try:
        if not os.path.exists(resume_folder):
            os.makedirs(resume_folder)
            return []
            
        resume_files = [
            os.path.join(resume_folder, f) 
            for f in os.listdir(resume_folder) 
            if f.endswith(('.pdf', '.txt'))
        ]
        
        docs = []
        for file in resume_files:
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file)
                else:
                    loader = TextLoader(file)
                file_docs = loader.load()
                
                # Extract text from all pages
                full_text = "\n".join([doc.page_content for doc in file_docs])
                
                # Extract candidate info from text
                candidate_info = extract_candidate_info(full_text)
                
                for doc in file_docs:
                    doc.metadata.update({
                        "candidate_name": candidate_info["name"],
                        "experience": candidate_info["experience"],
                        "skills": candidate_info["skills"],
                        "projects": candidate_info["projects"]
                    })
                    docs.append(doc)
                print(f"Loaded resume for: {candidate_info['name']}")
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
        return docs
        
    except Exception as e:
        print(f"Error in load_resumes: {str(e)}")
        return []

def build_resume_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def search_candidates(query, vectorstore, top_k=5, query_type="general"):
    """
    Search candidates based on query type
    query_type can be: "general", "skills", "experience"
    """
    results = vectorstore.similarity_search(query, k=top_k)
    
    if query_type == "skills":
        # Return just skills for a given name
        for doc in results:
            name = doc.metadata.get("candidate_name", "Unknown")
            skills = doc.metadata.get("skills", [])
            if name.lower() == query.lower():
                return {"name": name, "skills": skills}
        return None
        
    elif query_type == "names_only":
        # Return just names for a general query
        names = set()
        for doc in results:
            names.add(doc.metadata.get("candidate_name", "Unknown"))
        return list(names)
        
    else:
        # Return full candidate info
        candidates = []
        for doc in results:
            candidate = {
                "name": doc.metadata.get("candidate_name", "Unknown"),
                "experience": doc.metadata.get("experience", 0),
                "skills": doc.metadata.get("skills", []),
                "projects": doc.metadata.get("projects", [])
            }
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

def create_resume_qa_chain(vectorstore):
    """Create a QA chain for resume search"""
    
    # Initialize Gemini model for chat
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7
    )
    
    # Create template for better responses
    template = """You are a helpful HR assistant for a hackathon event.
    Use the following resume information to answer questions:
    {context}
    
    Question: {question}
    
    Give a clear, structured response:
    - For questions about specific skills, list relevant candidates
    - For questions about a person, show their skills and experience
    - Always include the candidate's name in the response
    - If information is not found, clearly state that
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    
    return qa_chain

# Initialize resume search right after the functions
try:
    resume_docs = load_resumes("resumes")
    if not resume_docs:  # Check if docs list is empty
        print("No resume files found in the resumes folder")
        resume_vectorstore = None
    else:
        resume_vectorstore = build_resume_vectorstore(resume_docs)
except FileNotFoundError:
    print("Resumes folder not found")
    resume_vectorstore = None
except Exception as e:
    print(f"Resume vectorstore error: {str(e)}")
    resume_vectorstore = None

# --- Environment Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# speech_client = speech.SpeechClient()

# --- Page Config ---
st.set_page_config(
    page_title="Event Bot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- Hero Section ---
st.markdown("""
<div style="
    display: flex; 
    justify-content: center; 
    align-items: center; 
    min-height: 220px; 
    margin-bottom: 30px;">
    <div style="
        background: linear-gradient(135deg, #333333, #555555); 
        color: white; 
        border-radius: 18px; 
        padding: 32px 40px 28px 40px; 
        box-shadow: 0 4px 24px rgba(0,0,0,0.10); 
        max-width: 600px; 
        width: 100%; 
        text-align: center;">
        <h1 style="font-size: 2.2em; margin-bottom: 8px;">Welcome to Event Bot AI</h1>
        <p style="font-size: 1.15em; margin-bottom: 18px;">Your Personal Hackathon Assistant</p>
        <a href="#" style="text-decoration: none;">
            <button style="background: white; color: #333333; padding: 10px 28px; font-size: 1em; border: none; border-radius: 7px; cursor: pointer; font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.07);">Let Get Started</button>
        </a>
    </div>
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
            <div style="font-size: 100px; text-align: center;" class="pulse-animation">&#128075;</div>
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Personalized Recommendations", 
            "📋 Event Details", 
            "💬 Ask Me Anything",
            "📝 Event Feedback",
            "🔎 Resume Search"
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
    <div style='font-size: 60px;'>&#127775;</div>
    <p>Your responses will help us improve future events!</p>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab5:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💬 Candidate Search Assistant")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            user_input = st.chat_input("Ask about candidates (e.g., 'Who works with Gen AI?')")
            
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                
                try:
                    if resume_vectorstore:
                        with st.chat_message("assistant"):
                            with st.spinner("Searching..."):
                                qa_chain = create_resume_qa_chain(resume_vectorstore)
                                response = qa_chain.run(user_input)
                                st.write(response)
                                st.session_state.chat_history.append(
                                    {"role": "user", "content": user_input}
                                )
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": response}
                                )
                    else:
                        st.error("Resume database not available. Please check if resumes are loaded correctly.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
            # Show example questions
            with st.expander("📝 Example Questions"):
                st.markdown("""
                Try asking:
                - Who has experience with Gen AI?
                - What are Vishvas's skills?
                - Tell me about candidates who know Python
                - What projects has Arjun worked on?
                """)
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
<h3>Can&apos;t find your name?</h3>
<p>If you've registered but your name isn't showing up, please visit the registration desk with your confirmation email.</p>
</div>
""", unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div class="footer">
    <p>&copy; 2025 Event Bot AI. All rights reserved. | <a href="https://example.com">Privacy Policy</a> | <a href="https://example.com">Terms of Service</a></p>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_resumes(resume_folder="resumes"):
    try:
        # Print debug info
        print(f"Looking for resumes in: {os.path.abspath(resume_folder)}")
        
        if not os.path.exists(resume_folder):
            print(f"Creating resumes folder at {resume_folder}")
            os.makedirs(resume_folder)
            return []
            
        resume_files = [
            os.path.join(resume_folder, f) 
            for f in os.listdir(resume_folder) 
            if f.endswith(('.pdf', '.txt'))
        ]
        
        print(f"Found {len(resume_files)} resume files: {resume_files}")
        
        docs = []
        for file in resume_files:
            name = os.path.splitext(os.path.basename(file))[0].replace("_", " ")
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file)
                else:
                    loader = TextLoader(file)
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["candidate_name"] = name
                    docs.append(doc)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
        return docs
        
    except Exception as e:
        print(f"Error in load_resumes: {str(e)}")
        return []

def build_resume_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def search_candidates(query, vectorstore, top_k=5):
    results = vectorstore.similarity_search(query, k=top_k)
    names = set()
    for doc in results:
        names.add(doc.metadata.get("candidate_name", "Unknown"))
    return list(names)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def create_resume_qa_chain(vectorstore):
    """Create a QA chain for conversational resume search"""
    
    template = """You are a helpful HR assistant for a hackathon. 
    Use the following resume information to answer questions:
    {context}
    
    Question: {question}
    
    Important guidelines:
    - Only use information from the provided resumes
    - If someone asks about specific skills, list candidates with those skills
    - If asked about a specific person, provide their details in a structured way
    - Keep responses concise and professional
    - Use bullet points for multiple items
    - If information is not in the resumes, say so clearly
    
    Response:"""
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        top_p=0.8,
        top_k=40
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={
            "prompt": prompt,
        }
    )
    
    return qa_chain

# --- Initialize Resume Search ---
try:
    resume_docs = load_resumes("resumes")
    if not resume_docs:  # Check if docs list is empty
        print("No resume files found in the resumes folder")
        resume_vectorstore = None
    else:
        resume_vectorstore = build_resume_vectorstore(resume_docs)
except FileNotFoundError:
    print("Resumes folder not found")
    resume_vectorstore = None
except Exception as e:
    print(f"Resume vectorstore error: {str(e)}")
    resume_vectorstore = None
