from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import Scripts.AngelPronationApp as AP
import os
from uuid import uuid4
import threading
import time

app = Flask(__name__)

DOWNLOAD_PATH = '/home/valeogamer/PycharmProjects/Valgus/App/static/temp/download/'
RESULT_FOLDER = '/home/valeogamer/PycharmProjects/Valgus/App/static/temp/result/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')


@app.route('/result')
def result():
    img_name = request.args.get('img_name', default='', type=str)
    return render_template('result.html', img_name=img_name)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        # return 'Изображение не найдено', 400
        return redirect("/")

    file = request.files['file']
    if file.filename == '':
        # return 'Пустое имя файла', 400
        return redirect("/")

    filename = str(uuid4()) + os.path.splitext(file.filename)[1]  # Создание уникального имени файла
    file_path = os.path.join(DOWNLOAD_PATH, filename)
    file.save(file_path)
    flag = AP.image_process(file_path, filename)
    if flag:
        return redirect(url_for('result', img_name=filename))
    else:
        return 'Ошибка обработки изображения', 500  # Ошибка сервера

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # threaded=True
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
