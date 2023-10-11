import os
import time
import uuid
import tinydb
import json
import requests
import threading

from threading import Thread
from tinydb import TinyDB, Query

from src.config import HOST


lock = threading.Lock()
searchLock = threading.Lock()


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MyDB(metaclass=Singleton):
    def __init__(self):
        self.db = TinyDB('db.json')

        self._runPullThread()

    def _runPullThread(self):
        def poll():
            query = Query()
            while True:
                with lock:
                    allWaiting = self.db.search(query.status=='waiting')
                if allWaiting: print('Number of waiting results:', len(allWaiting))
                for item in allWaiting:
                    self._pull(item['filePath'])
                time.sleep(1)
                if self.exit: break
        self.exit = False
        self.thread = Thread(target=poll)
        self.thread.start()

    def sendFile(self, fpath):
        print(f'Send file {fpath}')
        data = self._search(fpath)
        if data is None:
            fid = uuid.uuid4().hex
            data = {
                'filePath': fpath,
                'fid': fid,
                'status': 'waiting',
                'results': {},
            }
        else:
            data['status'] = 'waiting'
            fid = data['fid']

        url = f'{HOST}/upload/{fid}'
        files = {'file': open(fpath, 'rb')}
        try:
            ret = requests.post(url, files=files)
            if ret.status_code == 200:
                self._update(fpath, data)
                print('Send done')
                return True
            else:
                print('Send failed')
                return False
        except:
            print('Connection failed')
            return False

    def _search(self, fpath):
        query = Query()
        try:
            with lock:
                res = self.db.search(query.filePath==fpath)
            if res: return res[0]
            return None
        except:
            return None

    def _update(self, fpath, data):
        with lock:
            query = Query()
            self.db.upsert(data, query.filePath==fpath)

    def _pull(self, fpath):
        data = self._search(fpath)
        if data is None: return
        if data['status'] == 'done': return
        if data['status'] != 'waiting': return

        fid = data['fid']
        url = f'{HOST}/status/{fid}'
        response = requests.get(url)
        print('Pulling data from backend:', fpath)
        if response.status_code == 200:
            results = response.json()
            data['results'] = results.get('results', {})
            data['status'] = results.get('status', '')
            self._update(fpath, data)

    def getResults(self, fpath):
        data = self._search(fpath)
        if data is None: return {}
        return data

    def close(self):
        self.exit = True
        self.thread.join()


if __name__ == '__main__':
    db = MyDB()
    db.getResults('a')
