from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import Scripts.AngelPronationApp as AP
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
import io

app = Flask(__name__)


def visualization(Foot, left, right, apprx_l=False):
    """
    Визуализация
    """
    fig, ax = plt.subplots()
    ax.plot(left.x_top, left.y_top, 'r*')
    ax.plot(left.x_middle, left.y_middle, 'g*')
    ax.plot(left.x_bottom, left.y_bottom, 'r*')
    ax.plot([left.x_top, left.x_middle, left.x_bottom], [left.y_top, left.y_middle, left.y_bottom], '-ro')
    ax.plot(right.x_top, right.y_top, 'r*')
    ax.plot(right.x_middle, right.y_middle, 'g*')
    ax.plot(right.x_bottom, right.y_bottom, 'r*')
    ax.plot([right.x_top, right.x_middle, right.x_bottom], [right.y_top, right.y_middle, right.y_bottom], '-ro')
    if apprx_l:
        lw = 3
        ax.plot([left.x_up_l, left.x_down_l, left.x_middle - left.x_middle / 2],
                [left.y_up_l, left.y_down_l, left.y_middle], '-c*', linewidth=lw)
        ax.plot([left.x_up_r, left.x_down_r, left.x_middle],
                [left.y_up_r, left.y_down_r, left.y_middle], '-b*', linewidth=lw)
        ax.plot([abs((left.x_up_l + left.x_up_r) / 2), left.x_middle, left.x_down_l],
                [left.y_min, left.y_middle, left.y_middle], '-r^', linewidth=lw)
        ax.plot([left.x_up_l, left.x_down_l, left.x_middle - left.x_middle / 4],
                [left.y_up_l, left.y_down_l, left.y_middle], '-c*', linewidth=lw)
        ax.plot([left.x_up_r, left.x_down_r, left.x_middle],
                [left.y_up_r, left.y_down_r, left.y_middle], '-b*', linewidth=lw)
        ax.plot([abs((left.x_up_l + left.x_up_r) / 2), left.x_middle, left.x_down_l],
                [left.y_min, left.y_middle, left.y_middle], '-r^', linewidth=lw)
    ax.invert_yaxis()
    ax.imshow(Foot.image.copy())
    left_angl = Foot.angle_between_vectors(left.x_top, left.y_top, left.x_middle,
                                           left.y_middle, left.x_bottom, left.y_bottom)
    right_angl = Foot.angle_between_vectors(left.x_top, left.y_top, left.x_middle,
                                            left.y_middle, left.x_bottom, left.y_bottom)
    ax.text(left.x_middle, left.y_middle, f'{left_angl:.04}', fontsize=15, color='blue', ha='right')
    ax.text(right.x_middle, right.y_middle, f'{right_angl:.04}', fontsize=15, color='blue', ha='left')
    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close(fig)  # Закрытие фигуры для освобождения ресурсов
    return buffer


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
    return render_template('result.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     # Получаем изображение из запроса
#     img = request.files['file']
#     # Обработка изображения
#     foot, left, right = AP.image_process(img)  # Обратите внимание, что img передается в функцию image_process
#     img_buffer = visualization(Foot=foot, left=left, right=right)
#     # Отправляем изображение в браузер
#     return send_file(img_buffer, mimetype='image/png')

@app.route('/upload', methods=['POST'])
def upload():
    # Получаем изображение из запроса
    img = request.files['file']
    # Обработка изображения
    foot, left, right = AP.image_process(img)
    img_buffer = visualization(Foot=foot, left=left, right=right)
    # Сохраняем обработанное изображение
    img_buffer.seek(0)
    with open('static/temp/processed/processed_image.png', 'wb') as f:
        f.write(img_buffer.read())
    # Перенаправляем пользователя на страницу с результатами
    return redirect(url_for('result'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # threaded=True
