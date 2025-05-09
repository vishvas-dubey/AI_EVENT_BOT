import streamlit as st
import json
import os
from datetime import datetime
import re

st.set_page_config(page_title="Event Bot AI", layout="centered")
st.title("🤖 Event Bot AI – Your Hackathon Assistants for you")

# Load agenda and location info
with open("agenda.json", "r") as f:
    agenda = json.load(f)

with open("location.json", "r") as f:
    locations = json.load(f)

# Resume parser: extract keywords for session matching
def extract_keywords(text):
    return re.findall(r"\b(?:AI|ML|Python|Data|Cloud|LLM|Vision|NLP)\b", text, flags=re.IGNORECASE)

def match_sessions(keywords):
    matched = []
    for item in agenda["Day 1"]:
        for kw in keywords:
            if kw.lower() in item["session"].lower() and item not in matched:
                matched.append(item)
    return matched

# Feedback saver
def save_feedback(session, comment):
    feedback = {"timestamp": datetime.now().isoformat(), "session": session, "feedback": comment}
    if os.path.exists("feedback_log.json"):
        with open("feedback_log.json", "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(feedback)
    with open("feedback_log.json", "w") as f:
        json.dump(data, f, indent=2)

# Upload resume and show matched sessions
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
            st.markdown(f"- **{item['time']}** – {item['session']}")
    else:
        st.warning("No matching sessions found. Try using a more detailed resume.")

# Feedback collection
st.subheader("📝 Share Your Feedback")
session_name = st.selectbox("Select session you attended", [s["session"] for s in agenda["Day 1"]])
comment = st.text_area("How was the session?")
if st.button("Submit Feedback"):
    save_feedback(session_name, comment)
    st.success("Thanks! Your feedback was recorded.")

# Chat interface
st.subheader("💬 Ask Me Anything")
user_input = st.chat_input("Ask about agenda, location, lunch timing...")
if user_input:
    st.chat_message("user").write(user_input)
    response = ""

    if "agenda" in user_input.lower():
        response = "**Today's Agenda:**\n"
        for item in agenda["Day 1"]:
            response += f"- {item['time']}: {item['session']}\n"

    elif "washroom" in user_input.lower():
        response = f"The washroom is located: {locations['washroom']}"
    elif "lunch" in user_input.lower():
        response = f"Lunch is served: {locations['lunch']}"
    elif "helpdesk" in user_input.lower():
        response = f"The helpdesk is located: {locations['helpdesk']}"

    elif "time left" in user_input.lower() or "how much time" in user_input.lower():
        now = datetime.now()
        lunch_time = datetime(now.year, now.month, now.day, 13, 0, 0)
        diff = lunch_time - now
        if diff.total_seconds() > 0:
            minutes = int(diff.total_seconds() // 60)
            response = f"⏱️ {minutes} minutes left until lunch!"
        else:
            response = "🍽️ Lunch time has already started!"

    else:
        response = "Sorry, I can help with agenda, washroom, lunch time, feedback, and resume-based suggestions. Try again!"

    st.chat_message("assistant").write(response)