import pickle
from flask import Flask, request

app = Flask(__name__)

with open('static/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/classify')
def classify():
    text = request.args.get('text', '')
    return model.predict([text])[0]

if __name__ == '__main__':
    app.run(debug=True)
