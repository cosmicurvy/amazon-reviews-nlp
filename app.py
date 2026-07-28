from flask import Flask, render_template, request
from predict import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        review = request.form['review']

        result = predict_sentiment(review)

        return render_template("index.html", prediction= result, review = review, text_color = "purple")


if __name__ == "__main__":
    app.run(debug=True)