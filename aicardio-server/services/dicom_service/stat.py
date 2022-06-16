# coding=utf-8
import sys
import os
import glob
import pandas as pd
import time
import json
import logging

from model.user import User, UserSchema

_logger = logging.getLogger(__name__)


def get_relative_path(x, root):
	return x[len(root)+1:]


def get_device(x, root):
	rel = get_relative_path(x, root)
	# print(rel)
	return rel[:rel.find("/")]


def get_timestamp(x):
	return time.gmtime(os.path.getmtime(x))


def get_date(x):
	return time.strftime("%Y-%m-%d", x)


def get_user(user_map, x):
	return user_map[x] if x in user_map else 'unknown'


def check_annotated_frames(frames_point, frames_boundary):
	# print("fp: {} fb: {}".format(frames_point, frames_boundary))
	if (len(frames_point) != len(frames_boundary)):
		return 0

	cnt = 0
	for idx in range(len(frames_point)):
		# print(idx, len(frames_point[idx]) , len(frames_boundary[idx] ))
		# print(frames_boundary[idx])
		if len(frames_point[idx]) > 0 and len(frames_boundary[idx]) > 0:
			cnt += 1
	return cnt
	
def check_annotated_frames_ver04(data_all_frames):
	cnt = 0
	for idx, value in enumerate(data_all_frames):
		if "ef_point" in value and "ef_boundary" in value:
			points = value["ef_point"]
			boundary = value["ef_boundary"]
			if len(points) > 0 and len(boundary) > 0:
				# print("Frame: {} {} {}".format(idx, len(points), len(boundary)))
				cnt += 1
	return cnt


def count_annotated_frames(data_obj):
	cnt = 0
	if 'point' in data_obj and 'boundary' in data_obj:
		points = data_obj['point']
		boundary = data_obj['boundary']

		if 'frames' in points and 'frames' in boundary:
			frames_point = points['frames']
			frames_boundary = boundary['frames']
			cnt = check_annotated_frames(frames_point, frames_boundary)
	elif "dicomAnnotation" in data_obj:
		cnt = check_annotated_frames_ver04(data_obj["dicomAnnotation"])
	return cnt




def get_nchamber(data_obj):
	if "diagnosis" in data_obj:
		diagnosis = data_obj["diagnosis"]
		if "chamber" in diagnosis:
			return diagnosis["chamber"]
		
	if "dicomDiagnosis" in data_obj:
		diagnosis = data_obj["dicomDiagnosis"]
		if "chamber" in diagnosis:
			return diagnosis["chamber"]
	return "LABEL"


def get_nframe_nchamber(x):
	o = json.load(open(x, "r"))
	return count_annotated_frames(o), get_nchamber(o)


def get_user_map():
	users = User.query.all()
	user_map = {
		user.deviceID: user.fullname
		for user in users if user.deviceID is not None
	}
	return user_map


STATS_FOLDER = './data/stats'
ANNOTATION_FOLDER_DEFAULT = './data/json_data'


def stat_on_folder(force=False, dest_folder=ANNOTATION_FOLDER_DEFAULT):
	""" Description:
			if `force` is True, run stat on json folder
			if `force` is False, read stat data from existed csv files
		Return (dataframe, dataframe):
			stat data by overview and grouped by day and user
	"""
	if not os.path.exists(dest_folder):
		raise Exception("folder doesn't exist")
	if not os.path.exists(STATS_FOLDER):
		os.makedirs(STATS_FOLDER)

	if not force:
		# find existed stat file
		stat_dir = os.path.abspath(STATS_FOLDER)
		stats_file_list = glob.glob(os.path.join(stat_dir, 'stats*.csv'))
		day_man_stats_file_list = glob.glob(
			os.path.join(stat_dir, 'day_man_stats*.csv'))
		# force to run stat on json folder
		if len(stats_file_list) == 0 or len(day_man_stats_file_list) == 0:
			return stat_on_folder(force=True, dest_folder=dest_folder)
		# select lastest csv file to return stat data
		last_stats_file = max(stats_file_list, key=get_timestamp)
		df = pd.read_csv(last_stats_file, dtype={"device": str, "path": str})
		overview = df.to_dict('records')

		last_day_man_stats_file = max(
			day_man_stats_file_list, key=get_timestamp)
		df = pd.read_csv(last_day_man_stats_file, dtype={"device": str})
		by_date_and_user = df.to_dict('records')
		
	else:
		user_map = get_user_map()
		stats_file = '{}/stats_{}.csv'
		day_man_stats_file = '{}/day_man_stats_{}.csv'
		print("processing folder:", dest_folder)

		file_list = sorted(glob.glob(os.path.join(
			dest_folder, "**", "*.json"), recursive=True))
			
#         print("len file_list: {}".format(len(file_list)))
			
		# filter annotation files not in folder versions
		file_list = [x for x in file_list if x.find("/versions/") == -1]
		# convert relative path into absolute path
		
		file_list = [os.path.abspath(x) for x in file_list]
		dest_folder = os.path.abspath(dest_folder)

		# nframe = [get_nframe(x) for x in file_list]
		n_frame_n_chamber = [get_nframe_nchamber(x) for x in file_list]
		nframe = []
		nchamber = []
		for value in n_frame_n_chamber:
			nframe.append(value[0])
			nchamber.append(value[1])
		

		
		device = [get_device(x, dest_folder) for x in file_list]
		timestamp = [get_timestamp(x) for x in file_list]
		user = [get_user(user_map, d) for d in device]
		count = [1 for _ in file_list]
		df = pd.DataFrame.from_dict({
			'user': user,
			'device': device,
			'date': [get_date(x) for x in timestamp],
			'nchamber': nchamber,
			'nframe': nframe,
			'dicoms': count,
			'path': [get_relative_path(x, dest_folder) for x in file_list],
		})
		
#         print(df)
		
		now = time.strftime("%Y-%m-%d", time.gmtime(time.time()))
		df.to_csv(stats_file.format(STATS_FOLDER, now), index_label='#')
		# stat by man and day
		day_man_df = df.groupby(['date', 'user', 'device']).sum()
		day_man_df.to_csv(
			day_man_stats_file.format(STATS_FOLDER, now),
			index_label=['date', 'user', 'device']
		)
		# convert dataframe into table
		overview = df.reset_index().to_dict('records')
		by_date_and_user = day_man_df.reset_index().to_dict('records')

	return overview, by_date_and_user
