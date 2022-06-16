# coding=utf-8

import logging
from model import db, Basemodel, ma

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)
_round_number = 10


class Dicom(Basemodel):
    __tablename__ = 'dicoms'

    pathFile = db.Column(db.String(255), nullable=False)
    userID = db.Column(db.Integer)
    fileID = db.Column(db.String(255), nullable=False)
    nframe = db.Column(db.Integer)
    label = db.Column(db.Integer)
    deviceID = db.Column(db.String(50))
    lad = db.Column(db.Boolean, default=False)
    lcx = db.Column(db.Boolean, default=False)
    rca = db.Column(db.Boolean, default=False)
    isNotStandardImage = db.Column(db.Boolean, default=False)
    note = db.Column(db.Text)

    def __init__(self,
                 pathFile,
                 fileID=0,
                 userID=None,
                 label=None,
                 nframe=None,
                 deviceID=None,
                 lad=False,
                 lcx=False,
                 rca=False,
                 isNotStandardImage=False,
                 note=None):
        self.pathFile = pathFile
        self.fileID = fileID
        self.userID = userID
        self.nframe = nframe
        self.label = label
        self.deviceID = deviceID
        self.lad = lad
        self.lcx = lcx
        self.rca = rca
        self.isNotStandardImage = isNotStandardImage
        self.note = note

    @classmethod
    def save_data_annotate(self, **kwargs):
        valid_keys = ["pathFile", "label", "userID", "label", "nframe",
                      "deviceID", "lad", "lcx", "rca", "isNotStandardImage", "note"]
        dicom_obj = {key: value for key, value in kwargs.items() if key in valid_keys}
        try:
            dicom_ins = Dicom(**dicom_obj)
            db.session.add(dicom_ins)
            db.session.commit()
        except Exception as e:
            _logger.error(e)
            db.session.rollback()
