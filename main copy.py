from flask import Flask, request, make_response, redirect
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms.fields import StringField, PasswordField, SubmitField


app = Flask(__name__)
bootstrap = Bootstrap(app)

#rutas que son URL para el cliente

#esto es un decorador para decirle que la funcion hello es de la ruta es /, es pratcicamente un endpoint
@app.route("/")
def index():
    user_ip = request.remote_addr
    response = make_response(redirect("/show_information_address"))
    response.set_cookie("user_ip_information", user_ip)

    return response

@app.route("/show_information_address")
def show_information():
    user_ip = request.cookies.get("user_ip_information")
    return f"Hola como vas la ip es: {user_ip}"

app.run(host='0.0.0.0', port=81, debug=True)

class LoginForm(FlaskFrom):
    username = StringField ("Nombre del usuario")
    password = PasswordField("Contraseña")
    submit = SubmitField("Enviar datos")