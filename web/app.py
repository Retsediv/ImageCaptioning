from flask import Flask
from flask import request
from flask import render_template

import os

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=tmpl_dir)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getcaption', methods=['POST'])
def getcaption():
    print("data: ", request.args)
    # caption = "Test caption..."
    # return caption


    return "123"