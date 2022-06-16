import logging
from flask_restplus import Api
from flask import Blueprint, jsonify
# import config
from flask import Flask, request
import pymongo
from api import api_blueprint as app
import os
import json
import datetime

def converTimeFormat(timeString="20200805054351"):
    
    # TimeSave = time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(mtime))
    date_time = datetime.datetime.strptime(timeString, "%Y%m%d%H%M%S")
    return date_time.strftime("%Y/%m/%d %H:%M:%S") 


databaseClient = pymongo.MongoClient("localhost:27017")
aircardioDatabase = databaseClient["aicardio"]
dicomannotationCollection = aircardioDatabase["dicomannotation"]

ROOT_DATA_ANNOTATION = "/root/tuannm/dicom-server/data/json_data"
ROOT_DATA_MP4 = "/root/tuannm/data_representation/train"


ROUTE_DASHBOARD = {
    "study": "/study",
    "verification": "/verification",
    "quality": "/quality",
    "view": "/view"
}

def getNowTimeString():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")



@app.route(ROUTE_DASHBOARD["study"], methods=['GET'])
def get_case_id_list():
    # https://efb3f03d.ngrok.io/api/v1/study
    try:
#         results = dicomannotationCollection.find({}, {"IDStudy" : 1, "_id": 0})
#         results = [result['IDStudy'] for result in results]
        
        results = dicomannotationCollection.find({}, {"IDStudy" : 1, "StudyQuality": 1, "_id": 0}) # "IDStudy" : 1,
        dataCases = {}
        for case in results:
            note = case["StudyQuality"]["Note"]
            if note not in dataCases:
                dataCases[note] = []
            dataCases[note].append(case['IDStudy'])
            
        return jsonify(status=True, data=dataCases), 200

    except Exception as e:
        print("ERROR get_case_id_list: {}".format(e))
        return {'status': False, 'error': e}, 500

@app.route(f'{ROUTE_DASHBOARD["study"]}__num_total_cases', methods=['GET'])
def get_num_total_cases():
    # https://efb3f03d.ngrok.io/api/v1/study__num_total_cases
    try:
        results = dicomannotationCollection.find().count()
        print(results)
        # results = [result['IDStudy'] for result in results]
        
        return jsonify(status=True, data=results)

    except Exception as e:
        print("ERROR get_num_total_cases: {}".format(e))
        return {'status': False, 'error': e}, 500

@app.route(f'{ROUTE_DASHBOARD["study"]}__doctor_list', methods=['GET'])
def get_doctor_list():
    # https://efb3f03d.ngrok.io/api/v1/study__doctor_list
    try:
        # results = {}
        result = {
            "0339250262": "Vũ Thị Mai",
            "0339999191": "Đỗ Doãn Bách",
            "0968663886": "Lê Tuấn Thành",
            "0978380700": "Văn Đức Hạnh",
            "0978929599": "Nguyễn Đoàn Trung",
            "0963373013": "Trần Minh Phương"
        }

        # print(results)
        # results = [result['IDStudy'] for result in results]
        
        return jsonify(status=True, data=result), 200

    except Exception as e:
        print("ERROR get_doctor_list: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True


@app.route(f'{ROUTE_DASHBOARD["study"]}__hospital_list', methods=['GET'])
def get_hospital_list():
    # https://efb3f03d.ngrok.io/api/v1/study__hospital_list
    try:

        result = dicomannotationCollection.distinct("InstitutionName")
        

        # results = [result['IDStudy'] for result in results]
        
        return jsonify(status=True, data=result), 200

    except Exception as e:
        print("ERROR get_hospital_list: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True


@app.route(f'{ROUTE_DASHBOARD["study"]}__patient_name_list', methods=['GET'])
def get_patient_name_list():
    # https://efb3f03d.ngrok.io/api/v1/study__patient_name_list
    try:
        results = dicomannotationCollection.find({}, {"IDStudy" : 1, "_id": 0, "PatientName": 1})

        results = [result for result in results]


        print(results)
        # results = [result['IDStudy'] for result in results]
        
        return jsonify(status=True, data=results), 200

    except Exception as e:
        print("ERROR get_patient_name_list: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True


def get_annotation_patient_case(data):
    # calculate from Annotation DATA {"dicomAnnotation": {} }
    # print("get_annotation_patient_case")

    numberOfAnnotatedDicoms = 0
    numberOfAnnotatedFrames = 0

    numberOfAnnotations = 0


    numberOfAccepted = 0
    numberOfRejected = 0
    numberOfNotReview = 0
    
    lastReviewTime = ""
    lastAnnotationTime = ""

    viewsAvailable = {}

    listFileDicom = data["ListFileDicom"]
    # print("listFileDicom: {}".format(listFileDicom))
    # StudyInstanceUID = 
    studyInstanceUID = data["StudyInstanceUID"]

    doctorAnnotation = {}
    isDoneDataView = True
    isDoneDataAnnotation = True
    
    for idx, (key, file) in enumerate(listFileDicom.items()):
        viewFile = file.get("DicomView", {}).get("DataView", {}).get("View", "")
        if len(viewFile) == 0:
            isDoneDataView = False
            
        dicomAnnotation = file.get("DicomAnnotation", {})
#         dataStatistics = dicomAnnotation.get("DataStatistics", {})
#         verificationStatus = dicomAnnotation.get("Verification", {})
        
        # dicomView = {}
        # print(idx, file)

        if len(dicomAnnotation) > 0:
            numberOfAnnotatedDicoms += 1
            numFileFrames = 0
            listFrameAnnotated = []
            
            for history, value in dicomAnnotation.items():
                dataStatistics = value.get("DataStatistics", {}) 
                verification = value.get("Verification", {})
                if len(dataStatistics) > 0:
                    annotatedTime = dataStatistics["AnnotatedTime"]

                    lastAnnotationTime = max(lastAnnotationTime, annotatedTime)
                    annotatedHistory = history.split("____")
                    # print("history: {}".format(annotatedHistory))

                    relativePathAnnotationTime = f'{studyInstanceUID}____{dataStatistics["RelativePath"]}____{annotatedHistory[1]}'

                    doctorDeviceID = annotatedHistory[0]
                    # timeAnotatedHistory = 

                    if doctorDeviceID not in doctorAnnotation:
                        doctorAnnotation[doctorDeviceID] = {}


                    listFrameAnnotated += dataStatistics["ListFrameAnnotated"]

                    verificationStatus = verification.get("Status", "not_verification")
                    if verificationStatus != "not_verification":
                        isDoneDataAnnotation = False
                    
                    
                    # print("GO HERE {} {}".format(type(value), relativePathAnnotationTime))

                    verificationTime = verification.get("Time", "")

                    doctorAnnotation[doctorDeviceID][relativePathAnnotationTime] = {
                        "VerificationStatus": verificationStatus,
                        "VerificationTime": verificationTime
                    }

                    # print(doctorAnnotation)


                    if verificationStatus == "not_verification":
                        numberOfNotReview += 1
                    elif verificationStatus == "accepted":
                        numberOfAccepted += 1
                    elif verificationStatus == "rejected":
                        numberOfRejected += 1

                    # print("TUASFJSDJF")
                    # pass


        

            numberOfAnnotatedFrames += len(set(listFrameAnnotated))

        numberOfAnnotations += len(dicomAnnotation)

        # print("DICO")
        if "DicomView" in file:
            
            lastReviewTime = max(lastReviewTime, file["DicomView"].get("DataView", {}).get("LastViewTime", "")) # ["DataView"]["LastViewTime"])
            dicomView = file["DicomView"].get("DataView", {}).get("View", "")
            if type(dicomView) != str:
                dicomView = ""
                
            if dicomView not in viewsAvailable:
                viewsAvailable[dicomView] = 0
            viewsAvailable[dicomView] += 1
    # print(viewsAvailable)

    data["NumberOfAnnotatedDicoms"] = numberOfAnnotatedDicoms
    data["NumberOfAnnotatedFrames"] = numberOfAnnotatedFrames
    data["ViewsAvailable"] = viewsAvailable
    data["NumberOfAnnotations"] = numberOfAnnotations
    data["NumberOfAccepted"] = numberOfAccepted
    data["NumberOfRejected"] = numberOfRejected
    data["NumberOfNotReview"] = numberOfNotReview

    data["LastReviewTime"] = lastReviewTime
    data["LastAnnotationTime"] = lastAnnotationTime
    data["DoctorAnnotation"] = doctorAnnotation
    data["IsDoneDataView"] = isDoneDataView
    data["IsDoneDataAnnotation"] = isDoneDataAnnotation
    
    # print("END OF HERE")
    # data["DoctorAnnotation"] = {

    #     "0363839897": {
    #         "1.2.276.0.76.3.1.53.4903599.2.20150421143550.931.122466____1.2.840.113619.2.239.7255.1429611183.0.103.512____F97CAQ3A____20200808070125.000": {},
    #         "1.2.276.0.76.3.1.53.4903599.2.20150421143550.931.122466____1.2.840.113619.2.239.7255.1429611183.0.103.512____F97CAQ3A____20200808070126.000": {},
    #     },

    #     "0363839898": {
    #         "1.2.276.0.76.3.1.53.4903599.2.20150421143550.931.122466____1.2.840.113619.2.239.7255.1429611183.0.103.512____F97CAQ3A____20200808070125.000": {},
    #         "1.2.276.0.76.3.1.53.4903599.2.20150421143550.931.122466____1.2.840.113619.2.239.7255.1429611183.0.103.512____F97CAQ3A____20200808070126.000": {},
    #     }

    # }


    return data

@app.route(f'{ROUTE_DASHBOARD["study"]}__patient_case', methods=['GET'])
def get_patient_case():
    # https://efb3f03d.ngrok.io/api/v1/study__patient_case?IDStudy=000006
    try:
        args = request.args
        # print("args: {}".format(args))
        idStudy = args["IDStudy"]
        # print("idStudy: {}".format(idStudy))

        results = dicomannotationCollection.find({"IDStudy": idStudy}, {'_id': False})
        data = {}
        for result in results:
            # data[] = result
            # data = dict(result)
#             print(result)
            for k, v in result.items():
                # if k != "_id":
                data[k] = v
            # print(result)
        
        data = get_annotation_patient_case(data)

        
        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR get_patient_case: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True

@app.route(f'{ROUTE_DASHBOARD["study"]}__all_patient_case', methods=['GET'])
def get_all_patient_case():
    # https://efb3f03d.ngrok.io/api/v1/study__all_patient_case
    try:
        args = request.args
        # print("args: {}".format(args))
        # idStudy = args["IDStudy"]
        # print("idStudy: {}".format(idStudy))

        results = dicomannotationCollection.find({}, {'_id': False})
        data = []

        for result in results:

            datum = {}
            for k, v in result.items():
                # if k != "_id":
                datum[k] = v
            datum = get_annotation_patient_case(datum)


            data.append(datum)
        
        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR get_patient_case: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True

@app.route(f'{ROUTE_DASHBOARD["study"]}__patient_case_in_range', methods=['GET'])
def get_patient_case_in_range():
    # lay nhieu nhat 10 cases 1 lan
    # # https://efb3f03d.ngrok.io/api/v1/study__patient_case_in_range?from=000006&to=000010
    try:
        args = request.args
        # print("args: {}".format(args))

        idStudyFrom = int(args["from"])
        idStudyTo = min( int(args["to"]), idStudyFrom + 9)
        idStudyTo = max(idStudyTo, idStudyFrom)

        strStudy = []
        for idStudy in range(idStudyFrom, idStudyTo + 1):
            strStudy.append(f'{idStudy:06}')

        # print(strStudy)


        # if idStudyTo > idStudyFrom + 20:
        #     idStudyTo = 

        # print("idStudyFrom: {} idStudyTo: {}".format(int(idStudyFrom), int(idStudyTo)))

        results = dicomannotationCollection.find({"IDStudy": { "$in": strStudy } }, {'_id': False})

        data = []

        for result in results:
            datum = {}
            for k, v in result.items():
                datum[k] = v
            datum = get_annotation_patient_case(datum)

            data.append(datum)

        
        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR get_patient_case: {}".format(e))
        return {'status': False, 'error': e}, 500
    return True

@app.route(f'{ROUTE_DASHBOARD["study"]}__get_data_fields', methods=['GET'])
def get_data_fields():
    # https://efb3f03d.ngrok.io/api/v1/study__get_data_fields

    try:
        data = ['IDStudy', 'StudyInstanceUID', 'InstitutionName', 'PatientName', 'NumberOfDicoms', 'NumberOfFrames', 'ListFileDicom', 'NumberOfAnnotatedDicoms', 'NumberOfAnnotatedFrames', 'ViewsAvailable', 'NumberOfAnnotations', 'NumberOfAccepted', 'NumberOfRejected', 'NumberOfNotReview', 'LastReviewTime', 'LastAnnotationTime', 'DoctorAnnotation']
        return jsonify(status=True, data=data), 200
        # return 
    except Exception as e:
        print("ERROR get_patient_case: {}".format(e))
        return {'status': False, 'error': e}, 500

    
# man hinh 02 cua Le Linh, quick submit theo view DONE
@app.route(f'{ROUTE_DASHBOARD["view"]}__get_quick_all_view_study', methods=['GET'])
def get_quick_all_view_study():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # 68.183.186.28:5000/api/v1/view__get_quick_all_view_study?IDStudy=000002
    # return {
    #     "IDStudy": "000002",
    #     "RelativePath": "1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82",
    #     "View": "", ["2C", "3C", "4C", "PTS_S", "PTS_L", "CW", "TW", "other", ...]
    #     "LastViewTime": "2020/09/20 20:20:20",
    # }

    requestBody = request.args
    # print("requestBody: {}".format(requestBody))
    try:
        idStudy = requestBody["IDStudy"]
        resultsReturn = {}

        results = dicomannotationCollection.find({ "IDStudy": idStudy})
        
        try:
            for result in results:
                ListFileDicom = result["ListFileDicom"]
                for idx, (key, file) in enumerate(ListFileDicom.items()):
                    
                    relativePath = file["RelativePath"]
                    dicomView = file.get("DicomView", {}).get("DataView", {})
#                     print(idx, relativePath, dicomView)
                    data = {
                        "IDStudy": idStudy,
                        "RelativePath" : relativePath,
                        "View": dicomView.get("View", ""),
                        "LastViewTime": dicomView.get("LastViewTime", ""),
                        "Note": dicomView.get("Note", ""),
                        "Quality": dicomView.get("Quality", 0),
                        "Privacy": dicomView.get("Privacy", False),
                        "ClipColor": dicomView.get("ClipColor", False),
                    }
                    resultsReturn[relativePath] = data
#                     print("relativePath: {} data: {}".format(relativePath, data))
    #             for result in results:
    #                 dicomView = result["ListFileDicom"][fileDicomFilterReplace]["DicomView"]["DataView"]
    #                 # print("dicomView: {}".format(dicomView))
    #                 data["View"] = dicomView.get("View", "")
    #                 data["LastViewTime"] = dicomView.get("LastViewTime", "")
    #                 data["Note"] = dicomView.get("Note", "")
    #                 data["Quality"] = dicomView.get("Quality", 0)

        except Exception as ee:
#             print("ERROR get_quick_all_view_study ee: {}".format(ee))
            pass

        return jsonify(status=True, data=resultsReturn), 200

    except Exception as e:
        print("ERROR get_quick_all_view_study: {}".format(e))
        return {'status': False, 'error': e}, 500
    
    
# man hinh 02 cua Le Linh, quick submit theo view DONE
@app.route(f'{ROUTE_DASHBOARD["view"]}__get_quick_dicom', methods=['GET'])
def get_quick_view_dicom():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # https://efb3f03d.ngrok.io/api/v1/view__get_quick_dicom?IDStudy=000002&RelativePath=1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82
    # return {
    #     "IDStudy": "000002",
    #     "RelativePath": "1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82",
    #     "View": "", ["2C", "3C", "4C", "PTS_S", "PTS_L", "CW", "TW", "other", ...]
    #     "LastViewTime": "2020/09/20 20:20:20",
    # }

    requestBody = request.args
    # print("requestBody: {}".format(requestBody))
    try:
        idStudy = requestBody["IDStudy"]
        relativePath = requestBody["RelativePath"]

        data = {
            "IDStudy": idStudy,
            "RelativePath" : relativePath,
            "View": "",
            "LastViewTime": "",
            "Note": "",
            "Quality": 0,
            "Privacy": False,
            "ClipColor": False,
        }

        # print("idStudy: {}".format(idStudy))
        fileDicomFilterReplace = relativePath.replace(".","_")


        results = dicomannotationCollection.find({ "IDStudy": idStudy})
        
        try:
            for result in results:
                dicomView = result["ListFileDicom"][fileDicomFilterReplace]["DicomView"]["DataView"]
                print("get_quick_view_dicom", dicomView)
                # print("dicomView: {}".format(dicomView))
                data["View"] = dicomView.get("View", "")
                data["LastViewTime"] = dicomView.get("LastViewTime", "")
                data["Note"] = dicomView.get("Note", "")
                data["Quality"] = dicomView.get("Quality", 0)
                data["Privacy"] = dicomView.get("Privacy", False)
                data["ClipColor"] = dicomView.get("ClipColor", False)
                
                
        except Exception as ee:
            pass

        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR dicom_view_submit: {}".format(e))
        return {'status': False, 'error': e}, 500
    
# man hinh 02 cua Le Linh, quick submit theo view DONE
@app.route(f'{ROUTE_DASHBOARD["view"]}__quick_dicom_submit', methods=['POST'])
def quick_view_dicom_submit():
    
    # case_id: IDStudy, dicom_id: RelativePath

#     https://efb3f03d.ngrok.io/api/v1/view__quick_dicom_submit
#     {
#         "IDStudy": "000002",
#         "RelativePath": "1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82",
#         "View": "3C",
#         "Quality": 0,
#         "Privacy": True,
#         "ClipColor": True,
#         "Note": "",
#     }

    requestBody = request.get_json()
    print("quick_view_dicom_submit requestBody: {}".format(requestBody))

    try:
        idStudy = requestBody["IDStudy"]

        relativePath = requestBody["RelativePath"]

        fileDicomFilterReplace = relativePath.replace(".","_")
        studyFilter = { "IDStudy": idStudy}

        if "View" in requestBody:
            keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.View"
            valueViewUpdate = { "$set": { keyUpdateView: requestBody["View"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)

        if "Quality" in requestBody:
            keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.Quality"
            valueViewUpdate = { "$set": { keyUpdateView: requestBody["Quality"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)

        if "Note" in requestBody:
            keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.Note"
            valueViewUpdate = { "$set": { keyUpdateView: requestBody["Note"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)

        if "Privacy" in requestBody:
            keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.Privacy"
            valueViewUpdate = { "$set": { keyUpdateView: requestBody["Privacy"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)
            
        if "ClipColor" in requestBody:
            keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.ClipColor"
            valueViewUpdate = { "$set": { keyUpdateView: requestBody["ClipColor"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)
            

        keyUpdateView = f"ListFileDicom.{fileDicomFilterReplace}.DicomView.DataView.LastViewTime"
        valueViewUpdate = { "$set": { keyUpdateView: getNowTimeString() } }
        # valueViewUpdate = { "$set": { keyUpdateView: "2019/09/10 07:53:32" } }
        dicomannotationCollection.update_one(studyFilter, valueViewUpdate)
        

        
        return jsonify(status=True, data=requestBody), 200

    except Exception as e:
        print("ERROR dicom_view_submit: {}".format(e))
        return {'status': False, 'error': e}, 500
    
    

@app.route(f'{ROUTE_DASHBOARD["quality"]}__get_study', methods=['GET'])
def get_quality_study():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # https://efb3f03d.ngrok.io/api/v1/quality__get_study?IDStudy=000002
    # return {
    #     "IDStudy": "000002",
    #     "StudyQuality": {
    #         "Note": "",
    #         "Quality": "", in [1, 2, 3, 4, 5] star thay cho vinif, kc,...
    #         "LastViewTime": "2020/09/09 20:20:20"
    #     }         
    # }



    try:
        requestBody = request.args
        idStudy = requestBody["IDStudy"]

        data = {
            "IDStudy": idStudy,
            "StudyQuality" : {
                "Note": "",
                "Quality": 0,
                "LastUpdateTime": ""
            }
        }

        results = dicomannotationCollection.find({ "IDStudy": idStudy})
        try:
            for result in results:
                studyQuality =result["StudyQuality"]
                data["StudyQuality"]["Note"] = studyQuality.get("Note", "")
                data["StudyQuality"]["Quality"] = studyQuality.get("Quality", 0)
                data["StudyQuality"]["LastUpdateTime"] = studyQuality.get("LastUpdateTime", "")
        except Exception as ee:
            pass 
    
        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR quality_study_submit: {}".format(e))
        return {'status': False, 'error': e}, 500
    
# Le Linh submit man hinh 02 theo study
@app.route(f'{ROUTE_DASHBOARD["quality"]}__study_submit', methods=['POST'])
def quality_study_submit():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # https://efb3f03d.ngrok.io/api/v1/quality__study_submit
    # post {
    #     "IDStudy": "000002",
    #     "Rating": in [1, 2, 3, 4, 5] star thay cho vinif, kc,...
    #     "Note": ""http://68.183.186.28:8050/case_id=000022
    # }


    requestBody = request.get_json()

    try:
        print("quality_study_submit requestBody: {}".format(requestBody))

        idStudy = requestBody["IDStudy"]

        studyFilter = { "IDStudy": idStudy}
        keyUpdateView = f"StudyQuality"
        
        dataView = {
            "Note": requestBody.get("Note", ""),
            "Quality": requestBody.get("Quality", ""),
            "LastUpdateTime": getNowTimeString()
        }


        valueViewUpdate = { "$set": { keyUpdateView: dataView } }

        dicomannotationCollection.update_one(studyFilter, valueViewUpdate)
        
        return jsonify(status=True, data=requestBody), 200

    except Exception as e:
        print("ERROR quality_study_submit: {}".format(e))
        return {'status': False, 'error': e}, 500
    
@app.route(f'{ROUTE_DASHBOARD["verification"]}__history_from_dicom_id', methods=['GET'])
def get_history_from_dicom_id():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # https://efb3f03d.ngrok.io/api/v1/verification__history_from_dicom_id?IDStudy=000002&RelativePath=1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82
    
    try:
        args = request.args
        # print("args: {}".format(args))
        idStudy = args["IDStudy"]
        relativePath = args["RelativePath"]
        # print("idStudy: {} relativePath: {}".format(idStudy, relativePath))
        

        relativePathKey = relativePath.replace(".", "_")

        # DicomAnnotation = 


        results = dicomannotationCollection.find({"IDStudy" : idStudy})

        dataHistory = {}

        for result in results:
            # print("result: {}".format(result))
            fileAnnotation = result["ListFileDicom"][relativePathKey]

            dicomAnnotation = fileAnnotation["DicomAnnotation"]

            for history, annotation in dicomAnnotation.items():
                # print(history, annotation)
                
                splitHistory = history.split("____")

                doctorDeviceID = splitHistory[0]
                annotatedTime = splitHistory[1]
                # print("doctorDeviceID: {}".format(doctorDeviceID))
                # print("annotation = {}".format(annotation))

                studyInstanceUID = annotation["DataStatistics"]["StudyInstanceUID"]
                relativePath = annotation["DataStatistics"]["RelativePath"]

                annotationPath = os.path.join(ROOT_DATA_ANNOTATION, doctorDeviceID, f'{studyInstanceUID}____{relativePath}____{annotatedTime}.json')
                mp4Path = os.path.join(ROOT_DATA_MP4, f'{studyInstanceUID}/{relativePath}.mp4')
                # print("mp4Path: {}".format(mp4Path))

                annotatedData = {}
                if os.path.isfile(annotationPath):
                    with open(annotationPath, "r") as fr:
                        annotatedData = json.load(fr)

                dataHistory[history] = {
                    "Mp4File" : mp4Path,
                    "AnnotatedPath": annotationPath,
                    "AnnotatedData": annotatedData,
                    "VerificationStatus": annotation.get("Verification",{}).get("Status", "not_verification"),
                }



                # with open(annotationPath, "r")
        # results = [result['IDStudy'] for result in results]
        
        
        return jsonify(status=True, data=dataHistory), 200

    except Exception as e:
        print("ERROR get_history_from_dicom_id: {}".format(e))
        return {'status': False, 'error': e}, 500
    
    
# Phi submit du lieu veriy man hinh 03
@app.route(f'{ROUTE_DASHBOARD["verification"]}__annotation_submit', methods=['POST'])
def annotation_submit():
    
    # case_id: IDStudy, dicom_id: RelativePath
    
    # https://efb3f03d.ngrok.io/api/v1/verification__annotation_submit

    # POST {
    #     "IDStudy": "000002",
    #     "RelativePath": "1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82",
    #     "DoctorDeviceIDAndAnnotatedTime":  f'{phone}____{time}' "0968663886____20190910075332"
    #     "Status" : in ["verification_false", "verification_true", "not_verification"]
    #     "Rating" : in [1, 2, 3, 4, 5],
    #     "Note": "",
    # }


    requestBody = request.get_json()

    try:
        print("annotation_submit requestBody: {}".format(requestBody))

        idStudy = requestBody["IDStudy"]
        relativePath = requestBody["RelativePath"]
        doctorDeviceIDAndAnnotatedTime =requestBody["DoctorDeviceIDAndAnnotatedTime"]
        # verificationStatus = requestBody["VerificationStatus"]
        
        fileDicomFilterReplace = relativePath.replace(".","_")
        # fileDicomFilter = f'ListFileDicom.{fileDicomFilterReplace}.DicomView'
        
        studyFilter = { "IDStudy": idStudy}
        keyUpdateVerification = f"ListFileDicom.{fileDicomFilterReplace}.DicomAnnotation.{doctorDeviceIDAndAnnotatedTime}.Verification"
        
        dataVerification = {
            "Status":  requestBody.get("Status", "not_verification"),
            "Quality": requestBody.get("Quality", 0),
            "Note":  requestBody.get("Note", ""),
            "Time":  getNowTimeString(),
        }

        valueVerificationUpdate = { "$set": { keyUpdateVerification: dataVerification } }


        dicomannotationCollection.update_one(studyFilter, valueVerificationUpdate)


        return jsonify(status=True, data=requestBody), 200

    except Exception as e:
        print("ERROR annotation_submit: {}".format(e))
        return {'status': False, 'error': e}, 500

@app.route(f'{ROUTE_DASHBOARD["verification"]}__get_annotation_submit', methods=['GET'])
def get_annotation_submit():
    
    # case_id: IDStudy, dicom_id: RelativePath

    # https://efb3f03d.ngrok.io/api/v1/verification__get_annotation_submit?IDStudy=000002&RelativePath=1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82&DoctorDeviceIDAndAnnotatedTime=0968663886____20190910075332
    

    # return {
    #     "IDStudy": "000002",
    #     "RelativePath": "1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82",
    #     "DoctorDeviceIDAndAnnotatedTime":  f'{phone}____{time}' "0968663886____20190910075332"
    #     "Status" : in ["verification_false", "verification_true", "not_verification"]
    #     "Rating" : in [1, 2, 3, 4, 5],
    #     "Note": "",
    #     "Time": "2020/09/20 20:20:20"
    # }


    try:
        requestBody = request.args
        print("requestBody: {}".format(requestBody))

        idStudy = requestBody["IDStudy"]
        relativePath = requestBody["RelativePath"]
        doctorDeviceIDAndAnnotatedTime =requestBody["DoctorDeviceIDAndAnnotatedTime"]

        
        fileDicomFilterReplace = relativePath.replace(".","_")


        data = {
            "IDStudy": idStudy,
            "RelativePath": relativePath,
            "DoctorDeviceIDAndAnnotatedTime" : doctorDeviceIDAndAnnotatedTime,
            "Status": "",
            "Quality": 0,
            "Note": "",
            "Time": ""
        }
        # print("data: {}".format(data))

        results = dicomannotationCollection.find({ "IDStudy": idStudy})
        try:
            
            for result in results:
                verification = result["ListFileDicom"][fileDicomFilterReplace]["DicomAnnotation"][doctorDeviceIDAndAnnotatedTime]["Verification"]

                data["Status"] = verification.get("Status", "not_verification")
                data["Quality"] = verification.get("Quality", 0)
                data["Note"] = verification.get("Note", "")
                data["Time"] = verification.get("Time", "")
        except Exception as ee:
            pass
        

        return jsonify(status=True, data=data), 200

    except Exception as e:
        print("ERROR get_annotation_submit: {}".format(e))
        return {'status': False, 'error': e}, 500
    
    
class AnnotationDataLoader():
    
    def __init__(self):
        
        pass
    
    
    def getAnnotationDataFile(self, newDes, lastFileRemove, annotatedData):
        
        studyInstanceUID, relativePath, annotatedTime = self.__splitStudySopIUIDByFileName(newDes)


        annotatedTimeFormat = converTimeFormat(annotatedTime)
        
        view = self.__getView(annotatedData)

        doctorDeviceID = self.__getDoctorDeviceID(annotatedData)

        numberOfFrame, listFrameAnnotated, listFrameEndothelial, listFramePeripheral, listFrameEndothelialPeripheral = self.__getAnnotatedInformation(annotatedData)
        
        result = {
                        
            "StudyInstanceUID": studyInstanceUID,
            
            "RelativePath": relativePath,
            "AnnotatedTime": annotatedTimeFormat,
            
            
            "View": view, # 2C, 3C, 4C, ...
            "NumberOfFrame": numberOfFrame, # so frame cua file dicom
            "DoctorDeviceID": doctorDeviceID,
            
            
            "NumberOfAnnotatedFrame": len(listFrameAnnotated),
            
            "ListFrameAnnotated": listFrameAnnotated, # danh sach cac frame dc ve
            "NumberOfEndothelialFrames": len(listFrameEndothelial), # Noi mac, vien trong (EF)
            "ListFrameEndothelial": listFrameEndothelial,
            
            "NumberOfPeripheralFrames": len(listFramePeripheral), # Ngoai mac, vien ngoai (GLS)
            "ListFramePeripheral": listFramePeripheral,
            
            "NumberOfEndothelialPeripheralFrames": len(listFrameEndothelialPeripheral), # Ngoai mac, moi mac ngoai (GLS, EF)
            "ListFrameEndothelialPeripheral": listFrameEndothelialPeripheral,
                        
        }
        return annotatedTime, result
    
    def __splitStudySopIUIDByFileName(self, fileName):
        sIUID_SopIUID_ATime = fileName.split("/")[-1][:-5].split("____")
        StudyInstanceUID = sIUID_SopIUID_ATime[0]
        RelativePath = sIUID_SopIUID_ATime[1] + "____" + sIUID_SopIUID_ATime[2]
        AnnotatedTime = sIUID_SopIUID_ATime[3].split(".")[0]
        
        return StudyInstanceUID, RelativePath, AnnotatedTime
    
    
    def __getView(self, annotatedData):
        return annotatedData.get('dicomDiagnosis', {}).get('chamber', 'NO_LABEL')
    
    # def __getNumberOfFrame(self, annotatedData):
        # return len(annotatedData.get('dicomAnnotation', []))
    
    def __getDoctorDeviceID(self, annotatedData):
        return annotatedData.get("deviceID", "no_deviceID")
    
    def __getAnnotatedInformation(self, annotatedData):
        dicomAnnotation = annotatedData.get("dicomAnnotation", [])
        numberOfFrame = len(dicomAnnotation)
        listFrameAnnotated = []
        listFrameEndothelial = [] # noi mac, EF
        listFramePeripheral = [] # ngoai mac, GLS
        listFrameEndothelialPeripheral = [] # noi mac, ngoai mac, EF_GLS
        
        for index, data in enumerate(dicomAnnotation):
            ef_points = data['ef_point']
            gls_points = data['gls_point']
            ef_boundary = data['ef_boundary']
            gls_boundary = data['gls_boundary']

            if len(ef_points) > 0 or len(gls_points) > 0:
#                 total_annotated_frame.append(index)
                listFrameAnnotated.append(index)

            if len(ef_points) == 7 and len(ef_boundary) > 0:
                listFrameEndothelial.append(index)

            if len(gls_points) == 7 and len(gls_boundary) > 0:
                listFramePeripheral.append(index)

            if len(ef_points) == 7 and len(gls_points) == 7 and len(ef_boundary) > 0 and len(gls_boundary) > 0:
                listFrameEndothelialPeripheral.append(index)
                
        return numberOfFrame, listFrameAnnotated, listFrameEndothelial, listFramePeripheral, listFrameEndothelialPeripheral
                

    
    def updateDataMongo(self, newDes, lastFileRemove, annotatedData):

        studyInstanceUID, relativePath, annotatedLastTime = self.__splitStudySopIUIDByFileName(newDes)
        annotatedTime, dataAnnotation = self.getAnnotationDataFile(newDes, lastFileRemove, annotatedData)

        # print(annotatedTime, annotatedLastTime)
        # print(json.dumps(dataAnnotation, indent=2))

        # return
        studyInstanceUID = dataAnnotation["StudyInstanceUID"]
        relativePath = dataAnnotation["RelativePath"]
        doctorDeviceID = dataAnnotation["DoctorDeviceID"]
        
#             dicomAnnotationCollection.update
        
        studyFilter = { "StudyInstanceUID": studyInstanceUID}
    
#             print(relativePath.replace(".","_"))
        relativePath = relativePath.replace(".","_")
    
    
        keyUpdateAnnotation = f"ListFileDicom.{relativePath}.DicomAnnotation"




    
        keyAnnotation = f'{keyUpdateAnnotation}.{doctorDeviceID}____{annotatedTime}'
        valueAnnotation = { "$set": { keyAnnotation: {} } }
        
        dicomannotationCollection.update_one(studyFilter, valueAnnotation)
        
        
        keyAnnotation = f'{keyUpdateAnnotation}.{doctorDeviceID}____{annotatedTime}.DataStatistics'
        valueAnnotation = { "$set": { keyAnnotation: dataAnnotation } }
        dicomannotationCollection.update_one(studyFilter, valueAnnotation)

        
        keyAnnotation = f'{keyUpdateAnnotation}.{doctorDeviceID}____{annotatedTime}.Verification'
        dataVerification = {
            "Status": "not_verification",
            "Quality": 0,
            "Note": "",
            "Time": ""
        }

        valueAnnotation = { "$set": { keyAnnotation: dataVerification } }
        dicomannotationCollection.update_one(studyFilter, valueAnnotation)



        if "View" in dataAnnotation:
            keyUpdateView = f"ListFileDicom.{relativePath}.DicomView.DataView.View"
            valueViewUpdate = { "$set": { keyUpdateView: dataAnnotation["View"] } }
            dicomannotationCollection.update_one(studyFilter, valueViewUpdate)


        keyUpdateView = f"ListFileDicom.{relativePath}.DicomView.DataView.LastViewTime"
        valueViewUpdate = { "$set": { keyUpdateView: getNowTimeString() } }
        # valueViewUpdate = { "$set": { keyUpdateView: "2019/09/10 07:53:32" } }
        dicomannotationCollection.update_one(studyFilter, valueViewUpdate)


        # remove old value
        # keyAnnotation = f'{keyUpdateAnnotation}'
        try:
            studyInstanceUID, relativePath, annotatedLastTime = self.__splitStudySopIUIDByFileName(lastFileRemove)
            valueAnnotation = { "$unset": { f'{keyUpdateAnnotation}.{doctorDeviceID}____{annotatedLastTime}' : 1 } }
            # print(keyAnnotation, valueAnnotation)
            dicomannotationCollection.update(studyFilter, valueAnnotation)
        except Exception as e:
            print("Error remove: {}".format( f'{keyUpdateAnnotation}.{doctorDeviceID}____{annotatedLastTime}'))
            pass