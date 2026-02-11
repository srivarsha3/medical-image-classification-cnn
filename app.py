from flask import Flask, render_template, request
import os
from src.util import predict_pneumonia

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = predict_pneumonia(filepath)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

