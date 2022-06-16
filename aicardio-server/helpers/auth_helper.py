import jwt
import datetime
from flask import current_app as app, request
from helpers.error_handler_helper import HandledError


def generate_token(user):
    return jwt.encode(
        {
            'user': user,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=app.config.get('JWT_EXP_TIME')),
            'iat': datetime.datetime.utcnow(),
        },
        app.config.get('SECRET_KEY'), algorithm='HS256'
    ).decode()


def verify_token(token):
    try:
        payload = jwt.decode(token, app.config.get(
            'SECRET_KEY'), algorithms='HS256')
        return payload['user']
    except jwt.ExpiredSignature:
        raise HandledError(error_code=401, message='expired_token')
    except jwt.InvalidTokenError:
        raise HandledError(error_code=401, message='unauthorized')

def authorize():
    token = request.headers.get('Authorization')
    try:
        user = verify_token(token)
        print(user)
    except Exception as e:
        print(e)
