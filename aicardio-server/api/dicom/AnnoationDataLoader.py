import os
import glob
import json
import time
import datetime
import pymongo

databaseClient = pymongo.MongoClient("mongodb://localhost:27017/")

aicardioDatabase = databaseClient["aicardio"]

studyCollection = aicardioDatabase["study"] # collection = table
dicomannotationCollection = aicardioDatabase["dicomannotation"]


def getNowTimeString():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def readFileJSON(file):
    with open(file, "r") as fr:
        return json.load(fr)


def readFileJSONTimePrint(file):
    st_time = time.time()
    with open(file, "r") as fr:
        data = json.load(fr)
    ed_time = time.time()
    print("Time reading : {} {}".format(file, ed_time - st_time))
    return data

    
def writeDataToJSON(data, file, indent=False):
    
    with open(file, "w") as fw:
        if indent:
            json.dump(data, fw, indent=2)
        else:
            json.dump(data, fw)

    print("Done write to file: {}".format(file))
    
    
def viewData(data, numView=5):
    for idx, (k, v) in enumerate(data.items()):
        if idx == numView:
            break
        print(idx, k, v)

    
def getDataFromJSON(data, key, default="no"):
    
    if key in data: 
        return data[key]["value"] 
    else:
        return default
    
    
def set_copy_file_same_time(file_src, file_des):
    mtime = os.path.getmtime(file_src)
    atime = os.path.getatime(file_src)
    os.utime(file_des, (atime, mtime))
    
    
def getFiles(folder="/data.local/data/dicom_data_20200821", pattern="*.json"):
    files = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    return [f for f in files if os.path.isfile(f)]

def getFolders(folder="/data.local/data/dicom_data_20200821"):
    folders = [ os.path.join(folder, f) for f in os.listdir(folder)]
    return [f for f in folders if os.path.isdir(f)]

def getTimeForFile(filePath):
    mtime = os.path.getmtime(filePath)
    timeFile = AnnotatedTime.strftime('%Y%m%d%H%M%S.000', time.gmtime(mtime))
    return timeFile

def converTimeFormat(timeString="20200805054351"):
    
    # TimeSave = time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(mtime))
    date_time = datetime.datetime.strptime(timeString, "%Y%m%d%H%M%S")
    return date_time.strftime("%Y/%m/%d %H:%M:%S") 

DOCTOR_BY_DeviceID = {
    "0339250262": "Vũ Thị Mai",
    "0339999191": "Đỗ Doãn Bách",
    "0968663886": "Lê Tuấn Thành",
    "0978380700": "Văn Đức Hạnh",
    "0978929599": "Nguyễn Đoàn Trung",
    "0963373013": "Trần Minh Phương"
}

    

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


        # return True

    
    
    # 1. Thêm collection: studystatistics (chạy 5 phút thống kê/ lần)
    # 5. Thêm collection: studydiagnosis  (history diagnosi của bác sĩ, trạng thái check)
    # 2. Thêm collection: dicomview
    # 3. Thêm collection: dicomannotation
    # 4. Thêm collection: dicomdiagnosis (history diagnosi của bác sĩ, trạng thái check)
    # Lưu giữ statistics theo từng ngày ()
    
# if __name__ == "__main__":
    
#     annotatedData = readFileJSON("/home/tuan/Desktop/1.2.840.113619.2.300.7348.1565874381.0.181____1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82____20190910075332.json")

#     annotationDataLoader = AnnotationDataLoader()
#     annotationDataLoader.updateDataMongo(
#         "/home/tuan/Desktop/1.2.840.113619.2.300.7348.1565874381.0.181____1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82____20200910075332.json",
#         "/home/tuan/Desktop/1.2.840.113619.2.300.7348.1565874381.0.181____1.2.840.113619.2.300.7348.1565874381.0.188.512____J8FGOU82____20190910075332.json",
#         annotatedData)

#     annotationDataLoader.pushRawDataDicomAnnotation()
    
# #     annotationDataLoader.updateDataQualityStudy()
    
#     annotationDataLoader.pushDataDicomAnnotation()
# #     annotationDataLoader.getInforStudyQuality()
    
    