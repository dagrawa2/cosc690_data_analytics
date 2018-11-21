import json
import pickle

def save_json(D, path):
	with open(path, "w") as f:
		json.dump(D, f, indent=2)

def load_json(path):
	with open(path, "r") as f:
		D = json.load(f)
	return D

def save_pickle(D, path):
	with open(path, "wb") as f:
		pickle.dump(D, f)

def load_pickle(path):
	with open(path, "rb") as f:
		D = pickle.load(f)
	return D

def day_index(year, date):
	print(type(date))
	print(date)
	if month == 4: return day
	if month == 5: return 30+day
	if month == 6: return 61+day
