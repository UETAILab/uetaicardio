# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Model Runtime.
Entry point for the model runtime.
"""
import json
import os
import flask
from flask import request

from model_runtime import ModelRuntime
import torch

torch.set_num_threads(8)


app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

service = ModelRuntime()

def validate_metadata(metadata):
    metadata['frame_time'] = float(metadata['frame_time'])
    metadata['x_scale'] = float(metadata['x_scale'])
    metadata['y_scale'] = float(metadata['y_scale'])
    metadata['heart_rate'] = int(metadata['heart_rate'])
    metadata['window'] = int(metadata['window'])
    return metadata

@app.route("/query", methods=["POST"])
def transformation():
    f = request.files['file']
    metadata = request.form.to_dict()
    metadata = validate_metadata(metadata)
    print(metadata)
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'tmp', "temp.mp4")
    f.save(file_path)
    predictions = service.predict(file_path, metadata)
    result = json.dumps(predictions)
    return flask.Response(response=result, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run("localhost", 8080)
