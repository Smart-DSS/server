from flask import Response,Flask,jsonify
from flask_cors import CORS
import json
import os

cwd = os.getcwd()

app = Flask(__name__)
CORS(app)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

@app.route("/test")
def test():
    return Response('Server is working!',status=201,mimetype='application/json')

@app.route("/api/home",methods=['GET'])
# def return_home():
    # return jsonify({
    #     "message":"Hello world!"
    # })
    # return 
def get_json_data():
    file_path = cwd+'/exceed_count_data.json'  # Adjust the path if necessary
    json_data = read_json_file(file_path)
    return jsonify(json_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)