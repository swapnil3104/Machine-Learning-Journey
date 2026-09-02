from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("salary_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    experience = float(request.form["experience"])

    prediction = model.predict([[experience]])

    salary = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction=salary
    )


if __name__ == "__main__":
    app.run(debug=True)