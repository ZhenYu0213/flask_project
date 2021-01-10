from flask_wtf import FlaskForm
from wtforms import StringField
from flask_wtf.file import FileField, FileRequired
from wtforms.validators import DataRequired, length, EqualTo


class ImageInfoForm(FlaskForm):
    image = FileField('image', validators=[FileRequired()])
    # imageName = StringField('name', validators=[DataRequired()])


class VideoInfoForm(FlaskForm):
    video = FileField('video', validators=[FileRequired()])
