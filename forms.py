from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    customer_id = StringField('Customer ID', validators=[DataRequired()])
    submit = SubmitField('Login')
