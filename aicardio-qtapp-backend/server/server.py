import os
import redis
import uuid
import json

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify

from config import get_env_config
from redis_db import RedisDB
from echols.log import logger


CONFIG = get_env_config()
app = Flask(__name__)
rdb = RedisDB(CONFIG.REDIS_URL, CONFIG.REDIS_DB)


@app.route('/status/<iid>', methods=['GET'])
def get_iid_status(iid: str):
    out = rdb.get_iid_status(iid)
    print(out.keys())
    return jsonify(out)

@app.route('/clear', methods=['GET'])
def clear_queue():
    rdb.clear_queue()
    return 'OK'

@app.route('/upload/<iid>', methods=['POST'])
def upload_files(iid: str):
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    logger.info(f'New file incomming ID: {iid}; FILENAME: {filename}')
    fpath = os.path.join('server/database', f'{iid}_{filename}')
    fpath = os.path.abspath(fpath)
    uploaded_file.save(fpath)
    ret = rdb.add(iid, fpath, 0)
    return {'success': ret}


if __name__ == '__main__':
    app.run(host=CONFIG.HOST, port=CONFIG.PORT, debug=CONFIG.DEBUG)
