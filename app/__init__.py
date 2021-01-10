from flask import Flask, render_template, jsonify, request
from PIL import Image
from flask_cors import CORS
import darknet_video_custom
import threading
import time
import os
import app.SaveSystem as save
from app.form import ImageInfoForm, VideoInfoForm
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/img'
IMG_OUTPUT_FOLDER = './static/img/boundedImg'

app = Flask(__name__, template_folder='../templates',
            static_folder='../static')
app.config['SECRET_KEY'] = '123456'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def SaveImage(image, imageName):
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))


def SaveVideo(video, videoName):
    video.save(os.path.join(app.config['UPLOAD_FOLDER'], videoName))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/PostImage', methods=['POST'])
def PostImage():
    form = ImageInfoForm()
    if request.method == 'POST':
        if form.image.data != '':
            image = form.image.data
            imageName = secure_filename(image.filename)+".jpg"
            t = threading.Thread(target=SaveImage(
                image, imageName))
            t.start()
            t.join()
            print("image save done")
            print(UPLOAD_FOLDER+"/"+imageName)
            _yolo = darknet_video_custom.YOLO()
            _yolo.detect_image(UPLOAD_FOLDER+"/"+imageName,
                               IMG_OUTPUT_FOLDER+"/"+imageName)
            return jsonify({'errCode': 1, 'errMsg': 'success'})
        else:
            print("fail")
            return jsonify({'errCode': 0, 'errMsg': 'fail'})


@app.route('/PostVideo', methods=['POST'])
def PostVideo():
    form = VideoInfoForm()
    if request.method == 'POST':
        if form.video.data != '':
            video = form.video.data
            videoName = secure_filename(video.filename)+".mp4"
            t = threading.Thread(target=SaveVideo(video, videoName))
            t.start()
            t.join()
            print("video save done")
            _yolo = darknet_video_custom.YOLO()
            _yolo.detect_video(UPLOAD_FOLDER+"/"+videoName,
                               IMG_OUTPUT_FOLDER+"/"+videoName)
            return jsonify({'errCode': 1, 'errMsg': 'success'})
        else:
            return jsonify({'errCode': 0, 'errMsg': 'fail'})
