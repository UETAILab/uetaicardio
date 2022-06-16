# coding=utf-8
import logging
from flask import Blueprint

_logger = logging.getLogger(__name__)

view_blueprint = Blueprint('view', __name__, url_prefix='/', static_folder='test', static_url_path='/test')

from view.routes import routing_table
from view import admin
