from flask import Flask, request, jsonify
import pickle
import re
import nltk
import os
import logging
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('stopwords')
nltk.download('wordnet')

model_path = os.path.join('model', 'model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    logger.error("Model or vectorizer file not found. Please ensure both files are available.")
    raise FileNotFoundError("Model or vectorizer file not found. Please ensure both files are available.")

try:
    model = pickle.load(open(model_path, "rb"))
    vector = pickle.load(open(vectorizer_path, "rb"))
except Exception as e:
    logger.error(f"Error loading model or vectorizer: {str(e)}")
    raise


lemmatizer = WordNetLemmatizer()
stpwrds = stopwords.words('english')


def fake_news_det(news):

    news = re.sub(r'[^a-zA-Z\s]', '', news)
    news = news.lower() 
    words = nltk.word_tokenize(news)

    corpus = [lemmatizer.lemmatize(word) for word in words if word not in stpwrds]


    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)


    prediction = model.predict(vectorized_input_data)

    return prediction

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()
        news = data.get('news')

        if not news:
            return jsonify({"error": "No news content provided."}), 400

        
        pred = fake_news_det(news)

        
        if pred[0] == 1:
            result = "Prediction: This news looks like Fake News 🚫"
        else:
            result = "Prediction: This news looks like Real News ✅"

        return jsonify({"prediction": result, "status": "success"})

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return jsonify({"error": str(e), "status": "failure"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "status": "failure"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
