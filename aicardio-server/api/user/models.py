from api import api
from flask_restplus import fields

signup = api.model('signup_form', {
    'fullname': fields.String(required=True),
    'phone': fields.String(required=True),
    'password': fields.String(required=True),
    'email': fields.String(required=True),
    'deviceID': fields.String(),
    'job': fields.String(),
    'department': fields.String(),
    'organization': fields.String(),
    'address': fields.String(),
    'address': fields.String(),
    'userID': fields.Integer(),
    'activate': fields.Boolean(),
})

login = api.model('login_form', {
    'phone': fields.String(required=True),
    'password': fields.String(required=True),
})
