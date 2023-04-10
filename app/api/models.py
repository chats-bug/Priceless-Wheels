import os
import uuid
from typing import Optional, Type, get_type_hints

import pandas as pd
from pydantic import BaseModel, Field


def make_optional(
		include: Optional[list[str]] = None,
		exclude: Optional[list[str]] = None,
):
	"""Return a decorator to make model fields optional"""
	
	if exclude is None:
		exclude = []
	
	# Create the decorator
	def decorator(cls: Type[BaseModel]):
		type_hints = get_type_hints(cls)
		fields = cls.__fields__
		if include is None:
			fields = fields.items()
		else:
			# Create iterator for specified fields
			fields = ((name, fields[name]) for name in include if name in fields)
			# Fields in 'include' that are not in the model are simply ignored, as in BaseModel.dict
		for name, field in fields:
			if name in exclude:
				continue
			if not field.required:
				continue
			# Update pydantic ModelField to not required
			field.required = False
			# Update/append annotation
			cls.__annotations__[name] = Optional[type_hints[name]]
		return cls
	
	return decorator


def generate_guid_string() -> str:
	"""Generate a new GUID string"""
	return str(uuid.uuid4())


@make_optional()
class CarDetailsModel(BaseModel):
	Acceleration: float
	Alloy_Wheel_Size: float = Field(..., alias='Alloy Wheel Size')
	Cargo_Volume: str = Field(..., alias='Cargo Volume')
	City: str
	Color: str
	Displacement: float
	Doors: float
	Drive_Type: str = Field(..., alias='Drive Type')
	Front_Brake_Type: str = Field(..., alias='Front Brake Type')
	Front_Tread: float = Field(..., alias='Front Tread')
	Fuel_Suppy_System: str = Field(..., alias='Fuel Suppy System')
	Gear_Box: str = Field(..., alias='Gear Box')
	Kerb_Weight: float = Field(..., alias='Kerb Weight')
	Max_Power_At: float = Field(..., alias='Max Power At')
	Max_Power_Delivered: float = Field(..., alias='Max Power Delivered')
	Max_Torque_At: float = Field(..., alias='Max Torque At')
	Max_Torque_Delivered: float = Field(..., alias='Max Torque Delivered')
	No_of_Cylinder: float = Field(..., alias='No of Cylinder')
	Rear_Brake_Type: str = Field(..., alias='Rear Brake Type')
	Seats: float
	Steering_Type: str = Field(..., alias='Steering Type')
	Super_Charger: str = Field(..., alias='Super Charger')
	Top_Speed: float = Field(..., alias='Top Speed')
	Turbo_Charger: str = Field(..., alias='Turbo Charger')
	Turning_Radius: float = Field(..., alias='Turning Radius')
	Tyre_Type: str = Field(..., alias='Tyre Type')
	Valve_Configuration: str = Field(..., alias='Valve Configuration')
	Valves_per_Cylinder: float = Field(..., alias='Valves per Cylinder')
	Wheel_Base: float = Field(..., alias='Wheel Base')
	Width: float
	body: str
	comfort_features: str
	exterior_features: str
	fuel: str
	interior_features: str
	ip: float
	km_driven: float
	mileage_new: float
	model: str
	myear: float
	oem: str
	owner_type: str
	safety_features: str
	state: str
	top_features: str
	transmission: str
	utype: str
	variant: str

