from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import numpy as np

app = Flask(__name__)

# Load the saved model and tokenizer
MODEL_PATH = "./news_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get("text", "")

    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and predict
    inputs = tokenizer(news_text, truncation=True, padding=True, max_length=256, return_tensors="tf")
    logits = model(inputs).logits
    prediction = tf.argmax(logits, axis=1).numpy()[0]

    # Map back to labels (based on your mapping: 1=Fake, 0=Real)
    result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)