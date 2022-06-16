# coding=utf-8

import logging
from flask_bcrypt import generate_password_hash, \
    check_password_hash
from sqlalchemy import or_
from model import db, Basemodel, ma
from helpers.error_handler_helper import HandledError

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)
_round_number = 10


class User(Basemodel):
    __tablename__ = 'users'
    fullname = db.Column(db.String(
        length=100, collation='utf8mb4_general_ci'), nullable=False)
    phone = db.Column(db.String(255), primary_key=True,
                      unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    deviceID = db.Column(db.String(50))
    email = db.Column(db.String(255), nullable=False)
    job = db.Column(
        db.String(length=255, collation='utf8mb4_general_ci'))
    department = db.Column(
        db.String(length=255, collation='utf8mb4_general_ci'))
    organization = db.Column(
        db.String(length=255, collation='utf8mb4_general_ci'))
    address = db.Column(
        db.String(length=255, collation='utf8mb4_general_ci'))
    activate = db.Column(db.Boolean, default=False)

    def __init__(self,
                 email,
                 phone,
                 fullname,
                 password,
                 activate=True,
                 job=None,
                 address=None,
                 deviceID=None,
                 department=None,
                 organization=None):
        self.fullname = fullname
        self.phone = phone
        self.password = self._encrypt_password(password)
        self.deviceID = deviceID
        self.email = email
        self.job = job
        self.department = department
        self.organization = organization
        self.address = address
        self.activate = activate

    def is_active(self):
        if self.activate != 'active':
            raise ValueError(
                'User %s has not been activated by admin.' % (self.fullname))
        return True

    @staticmethod
    def _encrypt_password(password):
        pw_hashed = generate_password_hash(password, _round_number)
        pw_hashed_decoded = pw_hashed.decode("utf-8")
        return pw_hashed_decoded

    @staticmethod
    def _check_password(pw_hash, pw_raw):
        return check_password_hash(pw_hash, pw_raw)

    @classmethod
    def create(self, *args, **kwargs):
        if bool(self.query.filter_by(phone=kwargs['phone']).first()):
            raise HandledError(message='existed_phone_number', error_code=400)
        try:
            new_user = User(**kwargs)
            db.session.add(new_user)
            db.session.commit()
            return new_user
        except Exception as e:
            _logger.error(e)
            db.session.rollback()
            raise e


class UserSchema(ma.ModelSchema):
    class Meta:
        model = User
