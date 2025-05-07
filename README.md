# AI_EVENT_BOT

AI Workshop Event Bot
Solution for the AI Hackathon Challenge
This repository contains a complete solution for the "Build the Event Bot" AI Hackathon challenge, creating a smart assistant to enhance participant experience during the upcoming workshop on May 18th, 2025.
Overview
The Event Bot is a conversational AI assistant that helps workshop participants with real-time information, personalized recommendations, and feedback collection. It leverages Google Gemini for natural language understanding and generation.
Key Features

User Authentication

Verifies participant registration status
Personalized welcome messages


Resume Processing

Extracts skills and experience from user resumes
Provides personalized workshop recommendations based on resume content


Event Information

Full workshop agenda details
Real-time updates on current sessions
Location guidance (washrooms, tea area, etc.)
Time-aware responses (e.g., countdown to lunch)


Participant Networking

Identifies other participants with similar skills/interests
Facilitates connections among attendees


Feedback Collection

Conversational session feedback collection
Rating system with comments



Technical Architecture
Backend (Python Flask)

User Authentication System: Verifies registration against database
Resume Processing Module: Uses Google Gemini to extract skills and match with workshop content
Event Information System: Provides time-aware responses about the event
Recommendation Engine: Matches user skills with relevant workshops
Feedback Collection: Stores and processes session feedback

Frontend (HTML/CSS/JavaScript)

Responsive Web Interface: Works on both desktop and mobile devices
Chat Interface: Natural conversation with the Event Bot
Quick Access Buttons: For common queries
Current Session Display: Shows what's happening now
Feedback Collection Form: Star rating with comments

AI Integration (Google Gemini)

Natural Language Understanding: Processes user queries
Context-Aware Responses: Maintains conversation context
Resume Analysis: Extracts skills and experience from resumes
Matching Algorithm: Connects resumes with relevant workshops
Personalized Recommendations: Based on user profiles

Flow Diagram
User → Authentication → Resume Upload → Bot Interface
                                            ↓
                                      ┌─────┴─────┐
                                      ↓           ↓
                             Information      Feedback
                             Retrieval       Collection
                                      ↓           ↓
                                Personalized   Session
                               Recommendations Ratings
Implementation Details
User Identification & Registration

User enters their name
System verifies against registration database
If registered, proceeds to resume upload or bot interface
If not registered, directs to registration desk

Resume Processing

User uploads or pastes resume text
Google Gemini extracts key skills, technologies, and experience
System matches skills with workshop content
Personalized recommendations are generated

Event Information System
Handles queries about:

Workshop agenda and schedule
Current and upcoming sessions
Time until lunch
Location of facilities
WiFi and other logistics

Feedback Collection

Prompts for feedback after each session
Collects star rating (1-5)
Gathers detailed comments
Stores for analysis and improvement

Deployment Instructions

Clone this repository
Install required dependencies:
pip install -r requirements.txt

Set up your Google Gemini API key:
export GOOGLE_API_KEY="your_api_key_here"

Run the Flask application:
python app.py

Access the application at http://localhost:5000

Future Enhancements

Multi-language Support: Adding support for multiple languages
Speaker Profiles: Detailed information about presenters
Session Materials: Access to slides and resources
Follow-up Questions: Smart follow-ups based on previous interactions
Post-Event Survey: Comprehensive feedback collection after the event

Conclusion
This Event Bot solution addresses all six requirements from the hackathon challenge:

✅ Provides the meeting agenda
✅ Identifies participants with similar technical backgrounds
✅ Recommends relevant workshop sessions based on resumes
✅ Offers location guidance
✅ Provides real-time updates on event timing
✅ Collects conversational feedback after sessions

The implementation is fully functional, user-friendly, and provides significant value to workshop participants while demonstrating advanced AI capabilities using Google Gemini.
