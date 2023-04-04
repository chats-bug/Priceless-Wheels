import os
import datetime

from src.utils.constants import SAVE_DATE_TIME_FORMAT, RAW_DIR_PATH, RAW_SAVE_DATE_TIME_FORMAT


class DataLoader:
	def __init__(
			self,
			dir_path: str,
			raw_format: bool = False
	):
		self.dir_path = dir_path
		if self.dir_path == RAW_DIR_PATH or raw_format:
			self.save_date_time_format = RAW_SAVE_DATE_TIME_FORMAT
		else:
			self.save_date_time_format = SAVE_DATE_TIME_FORMAT
		self.latest_file = None
	
	def get_latest_file(self, begins_with: str, return_last: bool = False) -> str:
		# Return the latest file if it is already saved
		if self.latest_file and return_last:
			return self.latest_file
		
		# Get the list of files in the directory
		files = os.listdir(self.dir_path)
		
		# If the latest file is not found, then find it
		latest_file_time = None
		for file in files:
			if file.startswith(begins_with):
				# Get the file time
				file_time = datetime.datetime.strptime(
					file[len(begins_with) + 1:-4], self.save_date_time_format)
				if latest_file_time is None or latest_file_time < file_time:
					self.latest_file = file
					latest_file_time = file_time
		
		return self.latest_file
