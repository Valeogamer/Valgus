from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import Scripts.AngelPronationApp as AP
import os
from uuid import uuid4
import threading
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/PyProjects/Valgus/App/static/temp/uploads/'
PROCESSED_FOLDER = 'C:/PyProjects/Valgus/App/static/temp/processed/'

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
        return 'Изображение не найдено', 400

    file = request.files['file']
    if file.filename == '':
        return 'Пустое имя файла', 400

    filename = str(uuid4()) + os.path.splitext(file.filename)[1]  # Создание уникального имени файла
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    flag = AP.image_process(file_path, filename)
    if flag:
        return redirect(url_for('result', img_name=filename))
    else:
        return 'Ошибка обработки изображения', 500  # Ошибка сервера

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # threaded=True
