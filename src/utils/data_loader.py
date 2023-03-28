import os
import datetime

class DataLoader:
	def __init__(self, dir_path: str):
		self.dir_path = dir_path
		self.latest_file = None

	def get_latest_file(self, begins_with: str, override :bool = False) -> str:
		# Get the list of files in the directory
		files = os.listdir(self.dir_path)
		
		# Return the latest file if it is already saved
		if (self.latest_file is not None) and not override:
			return self.latest_file
		
		# If the latest file is not found, then find it
		latest_file_time = None
		for file in files:
			if file.startswith(begins_with):
				# Get the file time
				file_time = datetime.datetime.strptime(
					file[len(begins_with) + 1:-4], '%Y-%m-%d_%H-%M-%S')
				if latest_file_time is None or latest_file_time < file_time:
					self.latest_file = file
					latest_file_time = file_time

		return self.latest_file