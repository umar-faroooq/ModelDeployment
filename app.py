from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("model/cv.pkl", 'rb'))
clf = pickle.load(open("model/clf.pkl", 'rb'))

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email])
    predict = clf.predict(tokenized_email)
    predict = 1 if predict == 1 else -1



    return render_template("index.html", prediction=predict, email=email)

if __name__=="__main__":
    app.run(debug=True)
