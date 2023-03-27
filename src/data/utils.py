import os
import datetime

INDEX = "usedCarSkuId"

class Utility:
    # A utility function to get a number from a string
	def converst_to_number(x, conv: str = 'float'):
		x = str(x)
		new_str = ''
		is_dec = True
		for a in x:
			if 48 <= ord(a) <= 57:
				new_str += a
				continue
			elif a == ',' or a == '_':
				continue
			elif a == '.' and is_dec:
				new_str += a
				is_dec = False
			else:
				break
		
		if new_str == '':
			return None
		
		if conv == 'int':
			return int(new_str)
		
		return float(new_str)

	def get_begin_number(x):
		return Utility.converst_to_number(x, 'int')

	def get_begin_float(x):
		return Utility.converst_to_number(x, 'float')
	
	def get_latest_file(begins_with: str, dir_path: str = None) -> str:
		# Get the current directory
		dir_path = dir_path if dir_path is not None else os.getcwd()
		
		# Get the list of files in the directory
		files = os.listdir(dir_path)

		# Get the latest file
		latest_file = None
		latest_file_time = None
		for file in files:
			if file.startswith(begins_with):
				# Get the file time
				file_time = datetime.datetime.strptime(file[len(begins_with) + 1:-4], '%Y-%m-%d_%H-%M-%S')
				if latest_file_time is None or latest_file_time < file_time:
					latest_file = file
					latest_file_time = file_time

		return latest_file