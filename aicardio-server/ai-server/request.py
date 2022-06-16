import json

import requests

if __name__ == '__main__':
    metadata = {'siuid': '1.2.840.113663.1500.244401949268962451674789364464322408',
                'sopiuid': '1.2.840.113663.1500.246186806691199095732815633865241202',
                'frame_time': 0.019518, 'x_scale': 0.03075974343497136,
                'y_scale': 0.03075974343497136, 'heart_rate': "115", 'window': 29}
    files = {'file': open('./tmp/sample_dicom.mp4', 'rb')}
    r = requests.post("http://118.70.181.146:6060/query", files=files, data=metadata)
    print(r.content)