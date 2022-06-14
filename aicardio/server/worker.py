import os
import json
import time
import numpy as np

from config import get_env_config
from redis_db import RedisDB
from echols.main import Worker
from echols.log import logger


CONFIG = get_env_config()
rdb = RedisDB(CONFIG.REDIS_URL, CONFIG.REDIS_DB)
worker = Worker()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run(item):
    results = worker.run(item['dicom_path'])

    res_json_path = f"{item['dicom_path']}_res.json"
    #results = json.dumps(results, cls=NumpyEncoder)
    with open(res_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    item['results'] = results
    item['status'] = 'done'
    rdb.update(item)


if __name__ == '__main__':
    while True:
        item = rdb.next_request()
        if not item:
            print('Request Queue is empty!', end='\r')
            time.sleep(1)
            continue

        try:
            logger.info(f'Processing item: {item["iid"]}')
            run(item)
        except Exception as e:
            logger.info(f'Worker Error: {e}')
            item['status'] = 'error'
            rdb.update(item)
