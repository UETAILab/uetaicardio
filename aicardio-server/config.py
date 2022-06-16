import os
import logging
from dotenv import load_dotenv

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)

load_dotenv()

# URL
HOST_URL = 'http://0.0.0.0:5000'

# DIRECTORY
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = 'data'
UPLOAD_DIR = 'data/tmp'
GIF_DIR = 'data/gif_data'
JSON_DATA_DIR = 'data/json_data'
CONFIRM_DATA_DIR = 'data/confirm_data'


# DATABASE
_DB_NAME = os.getenv('DB_NAME', 'dicom_app')
_DB_USER = os.getenv('DB_USER', 'root')
_DB_PWD = os.getenv('DB_PWD', 'password')
_DB_HOST = os.getenv('DB_HOST', 'localhost')
_DB_PORT = os.getenv('DB_PORT', '3306')

# SQLALCHEMY
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://%s:%s@%s:%s/%s' % \
        (_DB_USER, _DB_PWD, _DB_HOST, _DB_PORT, _DB_NAME)
SQLALCHEMY_TRACK_MODIFICATIONS = False

ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'json', 'mp4'])

# UPLOAD FILES
MAX_CONTENT_LENGTH = 1000 * 1024 * 1024

SECRET_KEY = os.getenv('SECRET_KEY', "this_is_secret")
JWT_EXP_TIME = 24 * 3600

# ADMIN PASSWORD
ADMIN_PASSWORD = ['w3fBvU5YlWss', 'aicardio@ailab@uet']
