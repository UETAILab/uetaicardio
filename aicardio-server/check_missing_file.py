import os
import glob
import json
import shutil

ADD_FOLDER = "/media/tuan/DATA/AI-Cardio/JSON_DATA_V4/json/json_v4_20200909"

ROOT_FOLDER = "/media/tuan/DATA/AI-Cardio/JSON_DATA_V4/json/v4_path_train_20200820"



def copy_file_same_time(src_path, des_path):

    # print("COPY TO: {}".format(des_path))
    
    mtime = os.path.getmtime(src_path)
    atime = os.path.getatime(src_path)
    os.makedirs(os.path.dirname(des_path), exist_ok=True)
    shutil.copy(src_path, des_path)

    os.utime(des_path, (atime, mtime))

def move_file_same_time(src_path, des_path):

    # print("COPY TO: {}".format(des_path))
    
    mtime = os.path.getmtime(src_path)
    atime = os.path.getatime(src_path)
    os.makedirs(os.path.dirname(des_path), exist_ok=True)

    os.rename(src_path, des_path)

    os.utime(des_path, (atime, mtime))


def get_file_JSON(path_folder):
    json_files = glob.glob(os.path.join(path_folder, "**/*.json"), recursive=True)
    return sorted(json_files)

def CheckData(deviceID="0963373013"):
    cnt = 0
    add_path = os.path.join(ADD_FOLDER, deviceID)
    files = get_file_JSON(add_path)
    files = [[f, f[len(ADD_FOLDER) + 1: ]] for f in files]

    map_add = {}
    for file in files:
        
        f = file[1]
        ff = f[:-27]

        if ff not in map_add:
            map_add[ff] = []

        map_add[ff].append(file)


    # print(map_add)
    # return



    root_path = os.path.join(ROOT_FOLDER, deviceID)
    map_root = {}

    files = get_file_JSON(root_path)
    files = [[f, f[len(ROOT_FOLDER) + 1: ]] for f in files]
    for file in files:
        f = file[1]
        # print(f)
        # break
        ff = f[:-27]
        if ff not in map_root:

            map_root[ff] = []
        map_root[ff].append(file)
        # if f in map_add:
        #   print("file exist: {}")
    # print(files)

    for k, v in map_add.items():
        if k not in map_root:
            # print("Add to map_root {} -- {}".format(k,))
            src_json = v[0][0]
            des_json = src_json.replace("json_v4_20200909", "v4_path_train_20200820_v2")
            
            copy_file_same_time(src_json, des_json)
            # print("file exist: {}".format(os.path.isfile(des_json)))
            cnt += 1
            # print(src_json, des_json)

            # exit(0)

        else:
            # if len(map_root[k]) > 1:

            time1 = map_add[k][0][1][-23:]
            time2 = map_root[k][0][1][-23:]
            if time1 != time2:

                print(time1, "|", time2)

                print(map_root[k][0][0])

                src_json = v[0][0]
                des_json = src_json.replace("json_v4_20200909", "v4_path_train_20200820_v2")
                # print(src_json, des_json)

                copy_file_same_time(src_json, des_json)

                
                file_old = map_root[k][0][0]
                file_old_des = file_old.replace("v4_path_train_20200820", "v4_path_train_20200820_remove")
                print("file exist: {}".format(os.path.isfile(file_old)))
                move_file_same_time(file_old, file_old_des)
                print("file exist: {}".format(os.path.isfile(file_old)))
                

                # print("Replace file: {}")

            
            # pass
            # print("update {}".format(k))
    print("add: {}".format(cnt))

def walk_foler():
    files = get_file_JSON(ADD_FOLDER)
    map_checked = {}

    print(len(files))
    for idx, file in enumerate(files):
        if idx % 20 == 0:
            print("GO TO FILE: {}".format(idx + 1))
        # if idx < 112:
        #     continue
        with open(file, "r") as fr:
            data = json.load(fr)
        checked = data["checked"]

        if checked not in map_checked:
            map_checked[checked] = 0
        map_checked[checked] += 1
    print(map_checked)

# walk_foler()

CheckData("0339250262")
# CheckData("0339999191")
# CheckData("0963373013")
# CheckData("0968663886")
# print(len("____20200830141316.000.json"))