from __future__ import annotations

import logging
from typing import Any
import pandas as pd
import asyncio

from cardekho_api import get_car_details, CarDekhoAPIException


async def main():
    # Read the used card ID from a CSV file
    cars_list_df = pd.read_csv("../data/cardekho_all_cars.csv")
    used_car_ids = cars_list_df["usedCarSkuId"].tolist()
    all_car_details_df = pd.DataFrame()
    # Delete car_list_df to free up memory
    del cars_list_df

    # Create a function to get the car details
    async def get_transformed_car_details(used_car_id: str) -> dict[str, Any] | None:
        car = await call_cardekho_details_api(used_car_id)
        # print(f"Car details for {used_car_id} fetched successfully")
        return car_api_transform(car, used_car_id)

    count = 0
    failed = 0
    step_size = 200
    # Break the task into smaller tasks
    for i in range(0, len(used_car_ids), step_size):
        tasks = []
        for used_car_id in used_car_ids[i:i+step_size]:
            tasks.append(get_transformed_car_details(used_car_id))

        cars_details_tasks_result = await asyncio.gather(*tasks)
        cars_details = [
            car for car in cars_details_tasks_result if car is not None]
        count += len(cars_details)
        failed += len(tasks) - len(cars_details)

        # Append the details to a CSV file
        cars_details_df = pd.DataFrame(cars_details)
        all_car_details_df = pd.concat([all_car_details_df, cars_details_df])
        
        # cars_details_df.to_csv("../data/cardekho_all_cars_details.csv", mode="a", header=False, index=False)
        print(f"{len(cars_details)} successful, {len(tasks) - len(cars_details)} failed.  Total successful: {count}, total failed: {failed}. \t Dataframe size: {all_car_details_df.shape[0]}")

    # Do the remaining cars manually
    for used_car_id in used_car_ids[count:]:
        car = await get_transformed_car_details(used_car_id)
        cars_details = car_api_transform(car, used_car_id)
        if cars_details is None:
            continue
        cars_list_df = pd.DataFrame(cars_details)
        all_car_details_df = pd.concat([all_car_details_df, cars_details_df])
        # cars_list_df.to_csv("cardekho_all_cars_details_last.csv", header=True, index=False)
        count += 1

    all_car_details_df.to_csv("../data/car_details.csv", index=False)
    print(f"Exiting the program. Total cars fetched: {count}, total failed: {failed}")


async def call_cardekho_details_api(used_car_id: str) -> dict[str, Any] | None:
    try:
        car = await get_car_details(used_car_id)
        return car
    except CarDekhoAPIException as e:
        # Log the error in a file
        logging.basicConfig(
            filename='./logs/api_logs.log',
            filemode='w',
            format='%(name)s - %(asctime)s - %(levelname)s - %(message)s - %(data)s'
        )
        logging.error(e, extra={'data': e.data})
        return None


def car_api_transform(car: dict, used_car_id: str) -> dict[str, Any] | None:
    # The dictionary should be of the following format:
    # car = {
    # 	"overview": data["data"]["dataLayer"],
    # 	"features": data["data"]["carFeatures"],
    # 	"specifications": data["data"]["carSpecification"],
    # }

    try:
        # Transform the car details to the required format
        new_car = dict()
        new_car["usedCarSkuId"] = used_car_id
        # Get the features from the car details
        new_car["top_features"] = []
        for d in car["features"]["top"]:
            new_car["top_features"].append(d.get("value"))

        new_car["comfort_features"] = []
        new_car["interior_features"] = []
        new_car["exterior_features"] = []
        new_car["safety_features"] = []

        for d in car["features"]["data"]:
            if d["subHeading"] == "Comfort":
                new_car["comfort_features"] = [obj["value"]
                                               for obj in d["list"]]
            elif d["subHeading"] == "Interior":
                new_car["interior_features"] = [obj["value"]
                                                for obj in d["list"]]
            elif d["subHeading"] == "Exterior":
                new_car["exterior_features"] = [obj["value"]
                                                for obj in d["list"]]
            elif d["subHeading"] == "Safety":
                new_car["safety_features"] = [obj["value"]
                                              for obj in d["list"]]

        # Get the specifications from the car details
        for d in car["specifications"]["data"]:
            options_list = d["list"]
            for option in options_list:
                new_car[option["key"]] = option["value"]

        # Get the overview from the car details
        for key, value in car["overview"].items():
            new_car[key] = value

        return new_car

    except Exception as e:
        # Log the error in a file
        # logging.basicConfig(
        # 	filename='./logs/save_car_details_logs.log',
        # 	filemode='w',
        # 	format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
        # )
        # logging.error(e)
      #   print(f"Error while transforming car details: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())

# %%
