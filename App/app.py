from flask import Flask, render_template, request, redirect, url_for
import Scripts.AngelPronationApp as AP
import os
from uuid import uuid4

app = Flask(__name__)
# ubuntu
# DOWNLOAD_PATH = '/home/valeogamer/PycharmProjects/Valgus/App/static/temp/download/'
# RESULT_FOLDER = '/home/valeogamer/PycharmProjects/Valgus/App/static/temp/result/'

# windows
RESULT_FOLDER= 'C:/PyProjects/Valgus/App/static/temp/result/'
DOWNLOAD_PATH = 'C:/PyProjects/Valgus/App/static/temp/download/'


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
    name = request.args.get('name', default='', type=str)
    age = request.args.get('age', default='', type=int)
    foot_l = request.args.get('left_foot', default='', type=int)
    foot_r = request.args.get('right_foot', default='', type=int)
    return render_template('result.html', img_name=img_name, name=name, age=age, left_foot=foot_l, right_foot=foot_r)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect("/")

    file = request.files['file']
    name = request.form.get('name')
    age = request.form.get('age')

    if file.filename == '':
        return redirect("/")

    filename = str(uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(DOWNLOAD_PATH, filename)
    file.save(file_path)
    l_f, r_f = AP.image_process(file_path, filename)
    if l_f:
        return redirect(url_for('result', img_name=filename, name=name, age=age, left_foot=l_f, right_foot=r_f))
    else:
        return 'Ошибка обработки изображения', 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    # app.run(host='192.168.0.12', port=5000, debug=True, threaded=True)