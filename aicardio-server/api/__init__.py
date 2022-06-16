# coding=utf-8

import logging
from flask_restplus import Api
from flask import Blueprint, jsonify
import config

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)

api_blueprint = Blueprint(
    'blueprint',
    __name__,
    url_prefix='/api/v1'
)

api = Api(
    app=api_blueprint,
    title='Echocardio App Api',
    version='1.0'
)

from api.api_base import BaseApi
from api.user import api as api_user
from api.dicom import *

api.add_namespace(api_user)

# @api_blueprint.route("/test", methods=["GET"])
# def test():
#     res = stat.stat_on_folder()
#     # compressed_images = Compressor("./data/dicom_data", "F979KOS8").toJPEG()
#     return jsonify(res)
