from flask import Flask, request, jsonify, redirect, render_template_string
import joblib
import json

app = Flask(__name__)

# Load the trained model and vectorizer (corrected file name)
model = joblib.load("chatbot_model.pkl")  # Corrected to chatbot_model.pkl
vectorizer = joblib.load("vectorizer.pkl")

# Load the intents file
with open("intents.json", "r") as file:
    intents = json.load(file)

@app.route("/", methods=["GET"])
def index():
    # Simple HTML page with a button
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>College Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 20%; background-color: #f4f4f4; }
            button { 
                background-color: #007BFF; color: white; 
                border: none; padding: 15px 30px; font-size: 18px; 
                cursor: pointer; border-radius: 5px;
            }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Welcome to the College Chatbot</h1>
        <p>Click the button below to open the chatbot interface.</p>
        <form action="/redirect">
            <button type="submit">Go to Chatbot</button>
        </form>
    </body>
    </html>
    ''')

@app.route("/redirect", methods=["POST", "GET"])
def redirect_to_streamlit():
    # Redirect to Streamlit frontend
    return redirect("http://localhost:8501")  # Streamlit frontend URL

@app.route("/chat", methods = ["POST"])
def chatbot_response():
    #receive the user input and return bot responses
    try:
        data = request.get_json()
        user_message = data.get("message","")

        if not user_message:
            return jsonify({"responses": "No message provided"}), 400
        
        input_vector = vectorizer.transform([user_message])
        predicted_tag = model.predict(input_vector)[0]

        #Match prediction to response in intents file
        response_text = next((intent["responses"] for intent in intents["intents"] if intent["tag"] == predicted_tag), "Sorry, I don't understand that.")

        return jsonify({"responses": response_text})
    except Exception as e:
        return jsonify({"responses": f"Error: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
