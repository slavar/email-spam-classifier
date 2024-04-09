import json
from flask import Flask, jsonify, request
from app import cls

app = Flask(__name__)


@app.route('/')
def test():
    return 'test'


@app.route('/classify', methods=['POST'])
def classify():
    emails = request.json['emails']
    ret_val = json.dumps({'predictions': cls.predict(emails).tolist()})

    return jsonify(ret_val)
