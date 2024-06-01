from flask import Flask, request, jsonify
import util

app = Flask(__name__)
@app.route("/classify_image", methods=["GET", "POST"])
def classify_image():
    image = request.body["image"]
    response = jsonify(util.classify_image(image))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__=="__main__":
    print("Starting Python Flask Server For Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)