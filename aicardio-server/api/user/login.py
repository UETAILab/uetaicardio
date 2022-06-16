# coding=utf-8

import logging
from flask import request
from flask_restplus import Resource
from api.user import models as api_models, api
from api.api_base import BaseApi
from model.user import User as UserModel, UserSchema
from helpers.auth_helper import generate_token

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)

user_schema = UserSchema()


@api.route('/login')
class Login(Resource, BaseApi):
    @api.doc(description='Gửi username và password để đăng nhập')
    @api.expect(api_models.login)
    def post(self):
        body = request.json
        try:
            # find user by username
            user = UserModel.query.filter_by(phone=body['phone']).first()
            if not user:
                return self.api_response(error='account_not_existed', http_code=400)
            user = user_schema.dump(user).data
            if not user['activate']:
                return self.api_response(error='account_inactivate', http_code=400)
            # check password
            if not UserModel._check_password(
                pw_hash=user['password'],
                pw_raw=body['password']
            ):
                return self.api_response(error='wrong_password', http_code=400)
            token = generate_token(user)
            return self.api_response(data={'user': user, 'token': token})
        except ValueError as err:
            _logger.error(err)
            return self.api_response(error=str(err), http_code=400)
        except Exception as err:
            return self.api_response(http_code=500)
