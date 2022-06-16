import os
import json
import uuid
import redis


class RedisDB:

    qreqs = 'queue:requests'

    def __init__(self, url, db):
        self.db = redis.from_url(url=url, db=db)

    def next_request(self):
        data = self.db.blpop(self.qreqs, 1)
        if not data:
            return None
        data = json.loads(data[1])
        return data

    def add(self, iid, dicom_path, chamber=None):
        data = self._make_data(iid, dicom_path, chamber)
        self.db.rpush(self.qreqs, json.dumps(data))
        ret = self.update(data)
        return ret

    def update(self, new_data):
        iid = new_data.get('iid', None)
        if iid is None: return False
        try:
            if self.db.exists(iid):
                self.db.delete(iid)
            self.db.set(iid, json.dumps(new_data))
            return True
        except:
            return False

    def get_iid_status(self, iid):
        out = self.db.get(iid)
        if out is None: return {}
        out = out.decode('utf-8')
        out = json.loads(out)
        return out

    def clear_queue(self):
        n = self.db.llen(self.qreqs)
        for _ in range(n):
            self.db.blpop(self.qreqs)

    def _make_data(self, iid, dicom_path, chamber):
        #iid = uuid.uuid4().hex
        # status: error|waiting|done
        data = {
            'iid': iid,
            'dicom_path': dicom_path,
            'chamber': chamber,
            'status': 'waiting',
            'output_json': '',
        }
        return data
