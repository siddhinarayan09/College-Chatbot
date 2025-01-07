# College Chatbot

This repository contains the code for a chatbot that can answer various college-related queries using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The chatbot uses a Random Forest Classifier for intent classification, which is trained on a set of predefined intents.

## Features
- Answer common college-related questions such as:
  - College timings
  - Contact details
  - Courses and fees
  - Academic advising
  - Study abroad opportunities
  - Campus activities and more!

## Tech Stack
- Python
- Scikit-learn (for ML)
- Pandas
- Joblib (for model saving)
- Flask (for serving the chatbot)
- JSON (for storing intents data)

## Files Overview
- `intents.json`: Contains predefined questions and their respective responses categorized under different tags.
- `chatbot_model.pkl`: Saved Random Forest Classifier model.
- `vectorizer.pkl`: Saved TF-IDF vectorizer used for text vectorization.
- `chatbot.py`: The main Python script for chatbot training and interaction.
- `app.py`: The Flask app to deploy the chatbot.
- `README.md`: This file you are reading now.

## Installation

### Prerequisites
Ensure you have Python 3.x installed on your system.

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/college-chatbot.git
   cd college-chatbot
