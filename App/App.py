from flask import Flask, render_template, request, redirect, url_for, send_file
from DataBaseFoot import db, DBFootPronation
import AnglePronationApp as ap
from uuid import uuid4
import Config
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DBFoot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


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
    file_path = os.path.join(Config.DOWN_ABS_PATH, filename)
    file.save(file_path)
    l_f, r_f = ap.image_process(file_path, filename)

    if l_f is not None:
        # Сохранение данных в базе данных
        new_result = DBFootPronation(
            filename=filename,
            name=name, age=int(age),
            left_foot=l_f,
            right_foot=r_f,
            filename_download=Config.DOWN_PATH + filename,
            filename_unet_pred=Config.UNET_PATH + filename,
            filename_result=Config.RESULT_PATH + filename)
        db.session.add(new_result)
        db.session.commit()
        return redirect(url_for('result', img_name=filename, name=name, age=age, left_foot=l_f, right_foot=r_f))
    else:
        return 'Ошибка обработки изображения', 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, threaded=True)
    # app.run(host=Config.host, port=Config.port, debug=Config.debug, threaded=Config.thread)
