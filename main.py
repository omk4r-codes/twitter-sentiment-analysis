from flask import Flask, render_template, request, jsonify
import pickle

# Load the pre-trained model
with open('twitter_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Define a function to predict sentiment
def predict_sentiment(tweet):
    # Your preprocessing steps go here, like tokenization, vectorization, etc.
    # Assuming your model accepts a preprocessed input, so you just need to pass the tweet to it.
    sentiment = model.predict([tweet])
    return sentiment[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.json['tweet']
    sentiment = predict_sentiment(tweet)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
