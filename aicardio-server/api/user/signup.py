# coding=utf-8

import logging
from flask import request
from flask_restplus import Resource, fields
from api.api_base import BaseApi
from api.user import models as api_models, api
from model import db
from model.user import User as UserModel, UserSchema

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)

user_schema = UserSchema()

@api.route('/signup')
class Signup(Resource, BaseApi):
    @api.doc(description='Đăng ký tài khoản')
    @api.expect(api_models.signup, validate=True)
    def post(self):
        body = self._validate(request.json)
        try:
            new_user = UserModel.create(**body)
            return self.api_response(data=user_schema.dump(new_user).data)
        except Exception as e:
            _logger.error(e)
            return self.api_response(http_code=500, handled_error=e)

    @staticmethod
    def _validate(req):
        valid_keys = api_models.signup.keys()
        return {key: value for key, value in req.items() if key in valid_keys}
