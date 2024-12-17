import streamlit as st
import requests

st.title("College Chatbot")

#user input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        #send the request to flask api
        responses = requests.post("http://127.0.0.1:5000/chat", json = {"message": user_input})
        bot_response = responses.json().get("responses")

        #display the response
        st.text(f"Bot: {bot_response}")