import os
import sys
import random

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from models import CarDetailsModel

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir, os.pardir))
sys.path.insert(0, parent_dir_path)

app = FastAPI()
origins = [
	"http://localhost",
	"http://localhost:3000",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/")
async def hello():
	return {
		"message": "Welcome to Priceless Wheels API",
		"endpoints": {
			"/predict": "POST endpoint to predict the price of a car",
			"/predict_without_catboost": "POST endpoint to predict the price of a car without using CatBoost",
			"/sample": "GET endpoint to get a sample request body",
		},
	}


@app.get("/sample")
async def sample():
	# Load any model from the sample.json file
	sample_car = pd.read_json('./sample.json')
	
	return sample_car.to_dict(orient='records')[random.randint(0, 1000)]


@app.post("/predict")
async def predict(
		# The data will be in the body of the request
		car_details: CarDetailsModel = Body(...)
):
	print('Converting the class to a dataframe')
	# Convert the model to a dataframe
	car_details = pd.DataFrame([car_details.dict(by_alias=True)]).reset_index(drop=True)
	# Replace the empty strings with NaN
	car_details.replace('', 'nan', inplace=True)
	# print(car_details)
	
	print('Loading the models')
	# Load the model
	catboost_pipe = joblib.load('../../data/models/catboost_pipeline_2023-04-08_20-03-50.pkl')
	lightgbm_pipe = joblib.load('../../data/models/lightgbm_pipeline_2023-04-08_20-03-50.pkl')
	
	print('Predicting the price on - ')
	# Predict
	print('1. CatBoost')
	catboost_pred = np.exp(catboost_pipe.predict(car_details))
	print('2. LightGBM')
	lightgbm_pred = np.exp(lightgbm_pipe.predict(car_details))
	
	print('Averaging the predictions')
	# Average the predictions
	avg_pred = (catboost_pred + lightgbm_pred) / 2
	# Round the predictions to the nearest multiple of 1000
	avg_pred = np.round(avg_pred / 1000) * 1000
	print('Done')
	
	# Return the predictions
	return {"predictions": avg_pred.tolist()}


@app.post("/predict_without_catboost")
async def predict(
		# The data will be in the body of the request
		car_details: CarDetailsModel = Body(...)
):
	print('Converting the class to a dataframe')
	# Convert the model to a dataframe
	car_details = pd.DataFrame([car_details.dict(by_alias=True)]).reset_index(drop=True)
	# Replace the empty strings with NaN
	car_details.replace('', 'nan', inplace=True)
	# print(car_details)
	
	# Load the model
	lightgbm_pipe = joblib.load('../../data/models/lightgbm_pipeline_2023-04-08_20-03-50.pkl')
	
	print('Predicting the price on - ')
	# Predict
	print('1. LightGBM')
	lightgbm_pred = np.exp(lightgbm_pipe.predict(car_details))
	
	print('Averaging the predictions')
	# Average the predictions
	avg_pred = lightgbm_pred
	# Round the predictions to the nearest multiple of 1000
	avg_pred = np.round(avg_pred / 1000) * 1000
	print('Done')
	
	# Return the predictions
	return {"predictions": avg_pred.tolist()}
