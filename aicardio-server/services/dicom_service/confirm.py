import os
import json
import logging
from config import CONFIRM_DATA_DIR, JSON_DATA_DIR

_logger = logging.getLogger(__name__)


def get_confirm_on_file(device, filename):
    confirm_file_path = "%s.json" % os.path.abspath(
        os.path.join(*(CONFIRM_DATA_DIR, device, filename))
    )
    if os.path.exists(confirm_file_path):
        try:
            confirm_result = json.load(open(confirm_file_path, "r"))
            return confirm_result['checked']
        except Exception as e:
            _logger.error(e)
            return 0
    return 0


def set_confirm_on_file(device, filename, confirm, nchamber):
    confirm_file_path = os.path.abspath(os.path.join(CONFIRM_DATA_DIR, device))
    os.makedirs(confirm_file_path, exist_ok=True)
    try:
        fw = open('{}/{}.json'.format(confirm_file_path, filename), "w")
        if confirm:
            fw.write(json.dumps({'checked': 1, 'nchamber': nchamber}))
        else:
            fw.write(json.dumps({'checked': -1, 'nchamber': nchamber}))
        fw.close()
    except Exception as e:
        _logger.error(e)
        raise e
