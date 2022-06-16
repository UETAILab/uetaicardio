import os
import logging
from flask import current_app as app

_logger = logging.getLogger(__name__)


def get_gif_url(filename, deviceID):
    gif_dir = app.config.get('GIF_DIR', None)
    gif_abs_dir = os.path.abspath(gif_dir)
    gif_file_path = os.path.join(deviceID, "{}.gif".format(filename))

    if not os.path.exists(os.path.join(gif_dir, gif_file_path)):
        return None
    else:
        return os.path.join(gif_dir, gif_file_path)
