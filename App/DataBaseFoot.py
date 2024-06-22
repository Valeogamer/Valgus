from flask_sqlalchemy import SQLAlchemy
import datetime
db = SQLAlchemy()


class DBFootPronation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(100), unique=True, nullable=False)
    left_foot = db.Column(db.Integer, nullable=False)
    right_foot = db.Column(db.Integer, nullable=False)
    filename_download = db.Column(db.String(100), unique=True, nullable=False)
    filename_unet_pred = db.Column(db.String(100), unique=True, nullable=False)
    filename_result = db.Column(db.String(100), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now(), nullable=False)

    def __repr__(self):
        return f'<Result {self.filename}>'
