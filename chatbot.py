import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the intents data
with open("intents.json", "r") as file:
    intents = json.load(file)

# Flatten the data
data = []
for intent in intents['intents']:
    for pattern in intent['patterns']:  # Fixed to 'patterns'
        data.append((pattern, intent['tag']))

# Convert to dataframe
df = pd.DataFrame(data, columns=["patterns", "tag"])

# Text features and labels
x = df["patterns"]  # Fixed to 'patterns'
y = df["tag"]

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(x)

# Split into train-test sets
x_train, x_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Save the model and the vectorizer
import joblib
joblib.dump(rf_model, "chatbot_model.pkl")  # Save as .pkl
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved successfully!")
