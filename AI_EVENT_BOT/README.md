# AI Event Bot

This project is a Streamlit application designed to assist participants in a hackathon event. It provides information about the event schedule, important locations, and allows users to verify their registration through voice or text input. The application also features light and dark mode themes for better user experience.

## Project Structure

```
AI_EVENT_BOT
├── new
│   ├── app.py                # Main application file
│   ├── themes
│   │   ├── light.css         # CSS styles for light mode
│   │   └── dark.css          # CSS styles for dark mode
│   └── assets
│       └── custom.css        # Additional custom CSS styles
├── agenda.json               # Agenda data for the event
├── location.json             # Location data for event-related places
├── confirmed_users.json      # List of confirmed users for the event
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AI_EVENT_BOT
   ```

2. **Install Dependencies**
   Make sure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install streamlit speechrecognition streamlit-webrtc av
   ```

3. **Run the Application**
   Navigate to the `new` directory and run the Streamlit application:
   ```bash
   cd new
   streamlit run app.py
   ```

## Usage Guidelines

- Upon launching the application, users can verify their registration by speaking or typing their full name.
- The application displays the event agenda, important locations, and allows users to ask questions about the event.
- Users can switch between light and dark themes using the theme selection option.

## Files Overview

- **app.py**: The main application logic, handling user interactions and content display.
- **themes/light.css**: Styles for the light mode theme.
- **themes/dark.css**: Styles for the dark mode theme.
- **assets/custom.css**: Additional custom styles for further customization.
- **agenda.json**: Contains the schedule of events.
- **location.json**: Contains information about various locations relevant to the event.
- **confirmed_users.json**: Contains a list of users who are confirmed to attend the event.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.