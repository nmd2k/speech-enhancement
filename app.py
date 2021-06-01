import os
import argparse
from logging import debug
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, render_template
from model.config import UPLOAD_FOLDER

app = Flask(__name__, static_url_path='',
            static_folder='templates/static',
            template_folder='templates')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadfile', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']

    return render_template('upload_file.html')

if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

#     app.run(debug=True)
