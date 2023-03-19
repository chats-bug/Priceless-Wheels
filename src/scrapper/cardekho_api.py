from __future__ import annotations

from typing import Any

import httpx
from params import url_for_details, url as url_for_list

async def get_cars(url: str = None, query_params: dict = None) -> list[dict] | None:
	# We need a URL to make a request
	# The API structure needs a query parameter
	url = url or url_for_list
	if query_params is None:
		raise CarDekhoAPIException("No query parameters provided")
	
	try:
		async with httpx.AsyncClient() as client:
			response = await client.get(url, params=query_params)
			
			# If the request is successful, we get a 200 status code
			if response.status_code == 200:
				# print("Success")
				data = response.json()
				cars = data["data"]["cars"]
				return cars
			
			# !200 status code -> request failed
			else:
				failed_data = response.json()
				raise CarDekhoAPIException("Bad Request", failed_data)
	except Exception as e:
		raise CarDekhoAPIException("API Request failed ", e)


async def get_car_details(used_car_id: str) -> dict[str, Any]:
	# Used car ID is required to get the details
	if used_car_id is None:
		raise CarDekhoAPIException("No used car id provided")
	
	url = url_for_details
	query_params = {
		"city_id": "",
		"lang_code": "en",
		"regionId": 0,
		"otherinfo": "detailinfo",
		"usedcarid": used_car_id,
		# "device": "Web",
		# "pageType": "cls",
		# "devicePlatform": "pwa",
		# "pageLoadType": "client",
	}
	
	try:
		async with httpx.AsyncClient() as client:
			response = await client.get(url, params=query_params)
			
			# If the request is successful, we get a 200 status code
			if response.status_code == 200:
				# print("Success")
				data = response.json()
				car = {
					"overview": data["data"]["dataLayer"],
					"features": data["data"]["carFeatures"],
					"specifications": data["data"]["carSpecification"],
				}
				return car
			
			# !200 status code -> request failed
			else:
				failed_data = response.json()
				raise CarDekhoAPIException("Bad Request", failed_data)
	except Exception as e:
		raise CarDekhoAPIException("API Request failed ", e)


# Make an Exception class to handle errors
class CarDekhoAPIException(Exception):
	def __init__(self, message: str, data=None):
		self.message = message
		self.data = data
		super().__init__(self.message)
	
	def __str__(self):
		return f"{self.message}"
