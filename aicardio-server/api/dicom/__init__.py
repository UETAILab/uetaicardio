# coding=utf-8
import logging
import datetime
import urllib.parse
from flask import request, jsonify, send_from_directory, send_file
from model import Annotation
from api import api_blueprint as app
import json
import os
import glob
import zipfile
import io

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mongodb_api import *

_logger = logging.getLogger(__name__)

TIME_HOURS_REMOVE_FILE_JSON_VERSION = 48
ROOT_DICOM_REPRESENTATION = "/root/tuannm/data_representation/train"
ROOT_DIR_JSON = "/root/tuannm/dicom-server/data/json_data"
ROOT_DIR_ANNOTATION_CASE = "/root/tuannm/dicom-server/data/annotation_case"

ROOT_STUDY_QUALITY = "/root/tuannm/AICardio-Server/Data_Screen02"
ROOT_REPRESENTATION_RESIZE = "/root/tuannm/data_representation/resize"
ROOT_REPRESENTATION_FULL_SIZE = "/root/tuannm/data_representation/train"

def updataMongoDBAnnotation(newDes, lastFileRemove, dataAnnotation):
    annotationDataLoader = AnnotationDataLoader()
    annotationDataLoader.updateDataMongo(newDes, lastFileRemove, dataAnnotation)
#     return True

def getNameJSONFileByVersion(folder_path, pattern):
    newDes = os.path.join(folder_path, pattern)
    
    files = sorted(glob.glob(os.path.join(folder_path, f'*{pattern}*')), reverse=True)

    # print(len(files))
    # print(files[0])
    
    nowTime = datetime.datetime.now()
    nowTimeString = nowTime.strftime("%Y%m%d%H%M%S")
    newDes = f'{newDes}____{nowTimeString}.json'

    lastFileRemove = None
    deltaHours = None

    if len(files) > 0:
        lastFile = files[0]
        timeLastFile = lastFile.split("____")[-1][:-5]
        # print(timeLastFile)
        dtLastFile = datetime.datetime.strptime(timeLastFile, "%Y%m%d%H%M%S")
        # print(dt)
        # dt02 = datetime.datetime.strptime("20200908175654.000", "%Y%m%d%H%M%S.%f")
        

        deltaTime = nowTime - dtLastFile
        # print(dt02 - dt)
        deltaHours = int (deltaTime.total_seconds() / 60 / 60)
        # print(deltaHouse)

        if deltaHours < TIME_HOURS_REMOVE_FILE_JSON_VERSION: # time < 48h thi remove file cu, ghi de file moi
            lastFileRemove = files[0]

    # else:
    #     new_des = f'{new_des}____{nowTimeString}.json'

    return newDes, lastFileRemove, deltaHours




@app.route('/get_data_annotation_case', methods=['GET'])
def get_data_annotate_case():
    try:
        requestBody = request.args

        studyID = requestBody["StudyID"]
#         deviceID = requestBody["DeviceID"]
        deviceID = "0968663886"
        pattern = f'{deviceID}/{studyID}'

        files = sorted(glob.glob(os.path.join(ROOT_DIR_ANNOTATION_CASE, f'*{pattern}*')), reverse=True)
        data = {}
        if len(files) > 0:

            file_auto_analysis = files[0]
            print("Load data annotation case from file: {}".format(file_auto_analysis))
            with open(file_auto_analysis, 'r') as fr:
                data = json.load(fr)
            fr.close()
            return jsonify(status=True, data=data)
        return jsonify(status=True, data=data)
    except Exception as e:
        print("save_data_annotate_case Error: {}".format(e))
        return {'status': False, 'error': e}, 500



@app.route('/save_data_annotation_case', methods=['POST'])
def save_data_annotate_case():
    try:
        requestBody = request.get_json()
        studyID = requestBody["StudyID"]
        deviceID = requestBody["DeviceID"]

        nowTime = datetime.datetime.now()
        nowTimeString = nowTime.strftime("%Y%m%d%H%M%S")
        
        des_file = f'{deviceID}/{studyID}____{nowTimeString}.json'

        des_path = os.path.join(ROOT_DIR_ANNOTATION_CASE, des_file)
        os.makedirs(os.path.dirname(des_path), exist_ok=True)

        with open(des_path, "w") as fw:
                json.dump(requestBody, fw)

        print("Save data_annotate_case DONE: {}".format(des_path))

        return jsonify(status=True, data={"StudyID": studyID, "Time": nowTimeString})

    except Exception as e:
        print("save_data_annotate_case Error: {}".format(e))
        return {'status': False, 'error': e}, 500


@app.route('/save_data_annotate', methods=['POST'])
def save_data_annotate():
    req_body = request.get_json()
    if req_body is None or "path_file" not in req_body:
        return abort(400)
    file_name = req_body["path_file"]
    if '..' in file_name:
        return abort(400)
    

    if "get" in req_body:
        try:
#           print(req_body)
            sopiuid = req_body["sopiuid"]
            path_file = os.path.basename(req_body["path_file"])
            siuid = req_body["siuid"]

            file_auto_analysis = "/root/tuannm/dicom-server/data/json_machine/{}__{}__{}.json".format(siuid, sopiuid, path_file)
            
            print("Read Auto analysis file: {}".format(file_auto_analysis))
            _logger.info("Read Auto analysis file: {}".format(file_auto_analysis))

            with open(file_auto_analysis, 'r') as fr:
                data = json.load(fr)
            fr.close()

            return jsonify(status=True, data=data)
        except Exception as e:
            # _logger.info(f"GO TO get_auto_analysis: {req_body}")  
            _logger.error(e)
            return {'status': False, 'error': e}, 500
        

    try:
        
        datime = datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")

        file_dicom_dir = "/root/tuannm/dicom-server/data/json_data"

        try:
            # TODO
            dicom_tags = req_body["dicom_tags"]
            sopiuid = dicom_tags["(0008,0018)"]
            siuid = dicom_tags["(0020,000D)"]
            #     "StudyInstanceUID": "1.2.276.0.76.3.1.53.4903599.2.20150701171502.1280.130833",
            #      "SOPInstanceUID": "1.2.840.113619.2.239.3276.1435771044.0.4.512",

            req_body["dicom_tags"]["StudyInstanceUID"] = siuid
            req_body["dicom_tags"]["SOPInstanceUID"] = sopiuid
            
            req_body["checked"] = "not_check"
            
            basename = os.path.basename(req_body["path_file"])

            deviceID = req_body.get("deviceID", "no_device")
            folder_path = os.path.join(file_dicom_dir, deviceID)
            pattern_match = f'{siuid}____{sopiuid}____{basename}'
            
            des_path, file_remove, deltaHours = getNameJSONFileByVersion(folder_path, pattern_match)

            # des_path = os.path.join(file_dicom_dir, deviceID, f'{siuid}____{sopiuid}____{basename}____{datime}.json')
            
            os.makedirs(os.path.dirname(des_path), exist_ok=True)

            with open(des_path, "w") as fw:
                json.dump(req_body, fw)

            print("Save json_annotation DONE: {}".format(des_path))
            try:
                updataMongoDBAnnotation(des_path, file_remove, req_body)
            except Exception as ee:
                print("Exception remove db: {} -- {}".format(ee, des_path))
                
            
            if file_remove is not None and os.path.isfile(file_remove):
                os.remove(file_remove)
                print("deltaHours: {} Remove last version DONE: {}".format(deltaHours, file_remove))

            # return jsonify(status=True, data={})
            return jsonify(status=True, data=urllib.parse.urljoin(request.host_url, urllib.parse.quote(des_path[25:])))

        except Exception as e:
            
            file_path = req_body["path"].replace("/storage/emulated/0/Download/", "").replace("/", "__")
            deviceID = req_body.get("deviceID", "no_device")
            des_path = os.path.join(file_dicom_dir, deviceID, f'{file_path}____{datime}.json')
            
            print("Save json_dicom_annotation: {}".format(des_path))
            os.makedirs(os.path.dirname(des_path), exist_ok=True)

            
            with open(des_path, "w") as fw:
                json.dump(req_body, fw)

            # return jsonify(status=True, data={})
            return jsonify(status=True, data=urllib.parse.urljoin(request.host_url, urllib.parse.quote(des_path[25:])))

        except Exception as e:
            # _logger.info(f"GO TO get_auto_analysis: {req_body}")  
            _logger.error(e)
            print("ERROR: {}".format(e))
            return {'status': False, 'error': e}, 500

    except Exception as e:
        _logger.error(e)
        return {'status': False, 'error': e}, 500

@app.route('/get_file_zip_study_id', methods=['POST'])
def get_file_zip_study_id():
    req_body = request.get_json()
    # print("Start getZipStudyInstanceUID: {}".format(req_body))
    try:
        studyID = req_body["StudyID"]
        # TODO find StudyInstanceUID here
        # StudyInstanceUID = "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        StudyInstanceUID = req_body["StudyInstanceUID"]
#         print("get_file_zip_study_id studyID: {} StudyInstanceUID: {}".format(studyID, StudyInstanceUID))

        # SOPInstanceUID = req_body["SOPInstanceUID"]
        # relative_path = req_body["relative_path"]
        StudyFolder = os.path.join(ROOT_REPRESENTATION_RESIZE, StudyInstanceUID)

        lenStudyFolder = len(StudyFolder) + 1

        files = glob.glob(os.path.join(StudyFolder, "*jpg*"))

        # print(len(files))

        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w') as zf:

            for individualFile in files:
#                 print("individualFile: {} fn: {} StudyFolder: {} - {}".format(individualFile, individualFile[lenStudyFolder+1:], len(StudyFolder), lenStudyFolder))
                zf.write(individualFile, individualFile[lenStudyFolder:])


        memory_file.seek(0)
        return send_file(memory_file, attachment_filename=f'{studyID}.zip', as_attachment=True)


    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500


@app.route('/get_file_mp4_by_relative_path', methods=['POST'])
def get_file_mp4_by_relative_path():
    req_body = request.get_json()
    # print("Start getZipStudyInstanceUID: {}".format(req_body))
    try:
        StudyInstanceUID = req_body["StudyInstanceUID"] # "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        # SOPInstanceUID = req_body["SOPInstanceUID"]
        
        RelativePath = req_body["RelativePath"]

        FileMP4 = os.path.join(ROOT_REPRESENTATION_FULL_SIZE, StudyInstanceUID, RelativePath + ".mp4")

        return send_file(FileMP4, attachment_filename=f'{RelativePath}.mp4', as_attachment=True)


    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500
    
@app.route('/getZipStudyInstanceUID', methods=['POST'])
def getZipStudyInstanceUID():
    req_body = request.get_json()
    # print("Start getZipStudyInstanceUID: {}".format(req_body))
    try:
        StudyInstanceUID = req_body["StudyInstanceUID"] # "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        # SOPInstanceUID = req_body["SOPInstanceUID"]
        # relative_path = req_body["relative_path"]
        StudyFolder = os.path.join(ROOT_DICOM_REPRESENTATION, StudyInstanceUID)
        lenStudyFolder = len(StudyFolder) + 1

        files = glob.glob(os.path.join(StudyFolder, "*jpg*"))

        # print(len(files))

        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w') as zf:

            for individualFile in files:
            
                zf.write(individualFile, individualFile[lenStudyFolder+1:])


        memory_file.seek(0)
        print("getZipStudyInstanceUID {}".format(StudyInstanceUID))
        return send_file(memory_file, attachment_filename=f'{StudyInstanceUID}.zip', as_attachment=True)


    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500



@app.route('/getStudyQuality', methods=['POST'])
def getStudyQuality():
    req_body = request.get_json()

    # print("Start getZipStudyInstanceUID: {}".format(req_body))

    try:
        studyInstanceUID = req_body["StudyInstanceUID"] # "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        
        fileStudyQuality = os.path.join(ROOT_STUDY_QUALITY, studyInstanceUID + ".json")

        
        print("getStudyQuality file: {}".format(fileStudyQuality))

        with open(fileStudyQuality, 'r') as fr:
            data = json.load(fr)
        fr.close()
        
        return jsonify(status=True, data=data)

    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500

@app.route('/updateStudyQuality', methods=['POST'])
def updateStudyQuality():
    req_body = request.get_json()

    # print("Start getZipStudyInstanceUID: {}".format(req_body))

    try:
        StudyInstanceUID = req_body["StudyInstanceUID"] # "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        # reqData = req_body["data"]

        fileStudyQuality = os.path.join(ROOT_STUDY_QUALITY, StudyInstanceUID + ".json")


        
        print("updateStudyQuality file: {}".format(fileStudyQuality))
        data = {}
        
        try:
            with open(fileStudyQuality, 'r') as fr:
                data = json.load(fr)
            fr.close()

        except Exception as e:
            pass

        if "ListFileDicom" in req_body:

            for k1, v1 in req_body.items():
                data[k1] = v1
        else:

            listFileDicom = data["ListFileDicom"]
            for file in listFileDicom:
                if file["RelativePath"] == req_body["RelativePath"]:
                    file["FileNote"] = req_body.get("FileNote", file["FileNote"])
                    file["FileSlice"] = req_body.get("FileSlice", file["FileSlice"])
                    file["FileStarRating"] = req_body.get("FileStarRating", file["FileStarRating"])
        
        with open(fileStudyQuality, "w") as fw:
            json.dump(data, fw, indent=2)
        fw.close()

        return jsonify(status=True, data=urllib.parse.urljoin(request.host_url, urllib.parse.quote(fileStudyQuality)))
        # return jsonify(status=True, data=data)

    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500



@app.route('/getMP4StudySOPInstanceUID', methods=['POST'])
def getMP4StudySOPInstanceUID():
    req_body = request.get_json()
    # print("Start getZipStudyInstanceUID: {}".format(req_body))
    try:
        StudyInstanceUID = req_body["StudyInstanceUID"] # "1.2.276.0.76.3.1.53.4903599.2.20110311102146.6243.115722" # req_body["StudyInstanceUID"]
        # SOPInstanceUID = req_body["SOPInstanceUID"]
        
        RelativePath = req_body["RelativePath"]

        FileMP4 = os.path.join(ROOT_DICOM_REPRESENTATION, StudyInstanceUID, RelativePath + ".mp4")

        return send_file(FileMP4, attachment_filename=f'{RelativePath}.mp4', as_attachment=True)


    except Exception as e:
        print("ERROR: {}".format(e))
        return {'status': False, 'error': e}, 500


@app.route('/get_file_dicom', methods=['POST'])
def get_file_dicom():
    req_body = request.get_json()

    if "siuid" in req_body and "path_file" in req_body:
        try:
            path_file = req_body["path_file"]
            siuid = req_body["siuid"]

            file_dicom_dir = "/root/tuannm/dicom-server/data/dicom_data/{}".format(siuid)
            path = os.path.basename(path_file)

            # path = "K3AA0H02"
            print("GET DICOM file: {}".format(os.path.join(file_dicom_dir, path) ))
            
            return send_from_directory(file_dicom_dir, path, as_attachment=True)


        except Exception as e:
            # _logger.info(f"GO TO get_auto_analysis: {req_body}")  
            _logger.error(e)
            print("ERROR: {}".format(e))
            return {'status': False, 'error': e}, 500
        
    else:
        return {'status': False, 'error': e}, 500


@app.route('/get_file_zip_study', methods=['POST'])
def get_file_zip_study():
    req_body = request.get_json()

    if "siuid" in req_body:
        try:
            file_dicom_dir = "/root/tuannm/dicom-server/data/dicom_data"
            path = "{}.zip".format(req_body["siuid"])
            print("Download file zip study: {}".format(os.path.join(file_dicom_dir, path) ))
            return send_from_directory(file_dicom_dir, path, as_attachment=True)

        except Exception as e:
            # _logger.info(f"GO TO get_auto_analysis: {req_body}")  
            _logger.error(e)
            print("ERROR: {}".format(e))
            return {'status': False, 'error': e}, 500
        
    else:
        return {'status': False, 'error': e}, 500




    
@app.route('/verify_dicom_annotation', methods=['POST'])
def verify_dicom_annotation():
    req_body = request.get_json()

    # req_body = {
    #     "path": PATH_FILE, # "0339250262/1.2.840.113663.1500.1.467297889.1.1.20200623.91250.597____1.2.840.113663.1500.1.467297889.3.11.20200623.91909.791____IM_0011____20200826074842.000.json"
    #     "checked": "check_false_test"
    # }

    if "relative_path" in req_body and "checked" in req_body:

        try:

#             path = "{}".format(req_body["path"])

            relative_path = req_body["relative_path"]
#             print("Save req_body: {}".format(req_body))
        
            path_json = os.path.join(ROOT_DIR_JSON, relative_path)

            mtime = os.path.getmtime(path_json)
            atime = os.path.getatime(path_json)
            

            # print(os.path.isfile(path_json), path_json)
            with open(path_json, "r") as fr:
                data = json.load(fr)
            # print("checked: {}".format(data["checked"]))
            old_check = data.get("checked", "not_check")
            data["checked"] = req_body["checked"]
            data["checked_note"] = req_body
            
            print("Save verify_dicom_annotation from {} to {} --- path: {} checked_note: {}".format(old_check, req_body["checked"], path_json, data["checked_note"]))

#             file_back_up = path_json + ".sw"
#             os.rename(path_json, file_back_up)
            
#             with open(path_json, "w") as fw:
#                 json.dump(data, fw)
#             fw.close()

#             os.utime(path_json, (atime, mtime))

#             # with open(path_json, "r") as fr:
#             #     data = json.load(fr)
#             # print("new checked: {}".format(data["checked"]))

#             if os.path.isfile(file_back_up):
#                 os.remove(file_back_up)

            return jsonify(status=True, data=urllib.parse.urljoin(request.host_url, urllib.parse.quote(path_json)))


        except Exception as e:
            print("ERROR verify_dicom_annotation: {}".format(e))
            return {'status': False, 'error': e}, 500
        
    else:
        return {'status': False, 'error': e}, 500
    

