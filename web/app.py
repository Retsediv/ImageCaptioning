from flask import Flask
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename
from flask import jsonify
import os
import sys
print(os.getcwd())
sys.path.append("../")

from build_vocab import Vocabulary
import test


tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=tmpl_dir)
app.config['UPLOAD_FOLDER'] = './files'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getcaption', methods=['POST'])
def getcaption():
    print("data: ", request.form, request.files)
    # return "123"
    # file = request.files['file']
    # caption = "Test caption..."
    # return caption
    file = request.files.get("file")
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)
    # test.load_image(img_path)

    params = {'encoder_path':'../model/encoder-epoch-200-loss-0.0010738003766164184.ckpt',

    'decoder_path':'../model/decoder-epoch-200-loss-0.0010738003766164184.ckpt',

    'vocab_path':'../vocab.pkl',


'embed_size':256,
    'hidden_size':512,
    'num_layers':1}
    # return "2341"
    return jsonify(dict(text=str(test.main(image_path=img_path, **params))))

if __name__ == '__main__':
   app.run(debug = True)
