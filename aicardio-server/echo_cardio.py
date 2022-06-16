# coding=utf-8

import logging
import datetime
from flask import Flask, redirect, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from services import dicom_service

import model
from api import api_blueprint
from view import view_blueprint

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')

# config app from file config.py
app.config.from_pyfile("config.py", silent=True)

# databse setup
model.init_app(app)

# add blueprint apis
app.register_blueprint(api_blueprint)

# add blueprint web views
app.register_blueprint(view_blueprint)


# serve files
@app.route('/data/<path:filename>')
def download_file(filename):
    return send_from_directory(
        app.config['DATA_DIR'], filename, as_attachment=False)


# redirect to api page
@app.route('/api')
def redirect_to_blueprint():
    return redirect(api_blueprint.url_prefix)


# cross origin resource sharing
CORS(app, resources={r"/*": {"origins": "*"}})

# setup logging
logging.basicConfig(filename='vars/app.log', level=logging.DEBUG)
app.logger.handlers.extend(_logger.handlers)


# scheduling jobs
def job1():
#     print("RUN ECACH TIME: {}".format(datetime.datetime.now()))
    try:
        dicom_service.stat.stat_on_folder(force=True)
        print("Success Executed Stat at %s." % datetime.datetime.now())
        app.logger.info("Success Executed Stat at %s." %
                        datetime.datetime.now())
    except Exception as e:
        _logger.error(e)


scheduler = BackgroundScheduler()
scheduler.add_job(job1, 'interval', seconds=600)
scheduler.start()

# run
if __name__ == "__main__":
    app.run()
