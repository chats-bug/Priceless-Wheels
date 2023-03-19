from __future__ import annotations

import asyncio
import datetime
import logging

import pandas as pd

from cardekho_api import get_cars, CarDekhoAPIException
import params


async def call_cardekho_api(city_id: int, search_string: str, pageFrom: int, pagination: str, sort_order: str = "desc") -> dict | None:
    query_params = {
        "cityId": city_id,
        "lang_code": "en",
        "regionId": 0,
        "searchstring": search_string,
        "pagefrom": pageFrom,
        "sortby": "",
        "sortorder": {sort_order},
        "mink": "",
        "maxk": "",
        "dealer_id": "",
        "regCityNames": "",
        "regStateNames": "",
        "cellValue": "",
        "pagination": pagination
    }

    try:
        result = await get_cars(params.url, query_params)
        return result
    except CarDekhoAPIException as e:
        # Log the error in a file
        logging.basicConfig(filename='./logs/api_logs.log', filemode='w',
                            format='%(name)s - %(asctime)s - %(levelname)s - %(message)s - %(data)s')
        logging.error(e, extra={'data': e.data})
        return None


async def get_all_cars(city_id: int | None, search_string: str) -> list:
    count = 0
    pageFrom = 20
    best = 14
    normal = 0
    ftIndv = 0
    ftDl = 6
    ftPosMod = 0

    results = []

    for sort_order in ["desc", "asc"]:
        # Get the first batch of cars
        while True:
            pagination = '{"best":' + str(best) + ',"normal":' + str(normal) + ',"ftIndv":' + str(
                ftIndv) + ',"ftDl":' + str(ftDl) + ',"ftPosMod":' + str(ftPosMod) + '}'
            
            # Get the cars from the API
            result = await call_cardekho_api(city_id, search_string, pageFrom, pagination, sort_order=sort_order)
            
            # If there are no more cars, break the loop
            if result is None or len(result) == 0:
                break
            results.extend(result)  # Add the cars to the results
            # If there are less than 20 cars, that means we have reached the end of the list
            if len(result) < 20:
                break

            count += len(result)
            pageFrom += 20
            best += 20

        # Reset the parameters for the next batch of cars
        best = 439
        normal = 12
        ftIndv = 2
        ftDl = 7
        ftPosMod = 1

        while True:
            pagination = '{"best":' + str(best) + ',"normal":' + str(normal) + ',"ftIndv":' + str(
                ftIndv) + ',"ftDl":' + str(ftDl) + ',"ftPosMod":' + str(ftPosMod) + '}'

            # Get the cars from the API
            result = await call_cardekho_api(city_id, search_string, pageFrom, pagination, sort_order=sort_order)

            # If there are no more cars, break the loop
            if result is None or len(result) == 0:
                break
            results.extend(result)  # Add the cars to the results
            # If there are less than 20 cars, that means we have reached the end of the list
            if len(result) < 20:
                break

            count += len(result)
            pageFrom += 20
            normal += 15
            ftIndv = min(ftIndv + 1, 7)
            ftDl += 5

    print(f"Found {len(results)} cars in the base case {search_string}")
    return results


async def search_cars(
      cities: list[str] = None,
      brands: list[str] = None,
      fuel_types: list[str] = None,
      transmission_types: list[str] = None,
      body_types: list[str] = None,
      owners: list[str] = None,
) -> list[dict]:
        
    tasks = []
    all_cars = []
    print("Searching for cars in the following cases: ")
    print(f"Cities: {cities}")
    print(f"Brands: {brands}")
    print(f"Fuel Types: {fuel_types}")
    print(f"Transmission Types: {transmission_types}")
    print(f"Body Types: {body_types}")
    print(f"Owners: {owners}")
    print(f"Total number of cases: {len(cities) * len(brands) * len(fuel_types) * len(transmission_types) * len(body_types) * len(owners)}")
    
    print("\n\n------------------------------------\n\n")
    
    for city in cities:
        city_id = params.common_city_ids.get(city)
        city = city.replace(" ", "-").lower()

        for brand in brands:
            brand = brand.replace(" ", "-").lower()
            
            for fuel_type in fuel_types:
                fuel_type = fuel_type.replace(" ", "-").lower()
                
                for transmission_type in transmission_types:
                    transmission_type = transmission_type.replace(" ", "-").lower()
                    
                    for body_type in body_types:
                        body_type = body_type.replace(" ", "-").lower()
                        
                        for owner in owners:
                            owner = owner.replace(" ", "-").lower()
                            # used-cars+in+india+sedan+automatic+first-owner+honda+petrol
                            search_string = f"used-cars+in+{city}+{body_type}+{transmission_type}+{owner}+{brand}+{fuel_type}"
                            # all_cars.extend(await get_all_cars(city_id, search_string))
                            tasks.append(asyncio.create_task(get_all_cars(city_id, search_string)))
    
    all_cars_lists = await asyncio.gather(*tasks)
    
    # Flatten the list of lists into a single list of cars
    all_cars = [car for cars_list in all_cars_lists for car in cars_list]
    
    return all_cars


async def main():
    base_case_tasks = []
    
    # BASE CASES
    # Definition: Base cases are the cars less than 3500 in number in the Cardekho listing for the entire country
    # 1. All fuel types except Petrol and Diesel
    # 2. All body types except SUV, Sedan and Hatchback
    # 3. All car brands except Maruti, Hyundai, Honda, Mahindra, and Tata
    # 4. All base case cities defined in params.py
    non_base_case_fuel_types = ["Petrol", "Diesel"]
    base_case_fuel_types = [fuel for fuel in params.fuel_types if fuel not in non_base_case_fuel_types]
    non_base_case_body_types = ["SUV", "Sedan", "Hatchback"]
    base_case_body_types = [body for body in params.body_types if body not in non_base_case_body_types]
    non_base_case_brands = ["Maruti", "Hyundai", "Honda", "Mahindra", "Tata"]
    base_case_brands = [brand for brand in params.brands if brand not in non_base_case_brands]
    base_case_cities = params.common_base_cities
    non_base_case_cities = [city for city in params.common_cities if city not in base_case_cities]
    
    async def get_and_save_base_case_cars():
        print("Base Cases: ")
        print("Fuel Types: ", base_case_fuel_types)
        print("Body Types: ", base_case_body_types)
        print("Brands: ", base_case_brands)
        print("Cities: ", base_case_cities)
        
        # Get all the cars except the base cases
        
        # 1. Get the except base case fuel types
        for fuel_type in base_case_fuel_types:
            fuel_type = fuel_type.replace(" ", "-").lower()
            base_case_tasks.append(asyncio.create_task(get_all_cars(
                city_id=None,
                # Sample search string: used-cars+in+india+petrol
                search_string=f"used-cars+in+india+{fuel_type}"
            )))
        
        # 2. Get the except base case body types
        for body_type in base_case_body_types:
            body_type = body_type.replace(" ", "-").lower()
            base_case_tasks.append(asyncio.create_task(get_all_cars(
                city_id=None,
                # Sample search string: used-cars+in+india+muv
                search_string=f"used-cars+in+india+{body_type}"
            )))
        
        # 3. Get the except base case brands
        for brand in base_case_brands:
            brand = brand.replace(" ", "-").lower()
            base_case_tasks.append(asyncio.create_task(get_all_cars(
                city_id=None,
                # Sample search string: used-cars+in+india+maruti
                search_string=f"used-cars+in+india+{brand}"
            )))
        
        # 4. Get the base case cities
        for city in base_case_cities:
            city_id = params.common_city_ids.get(city)
            city = city.replace(" ", "-").lower()
            base_case_tasks.append(asyncio.create_task(get_all_cars(
                city_id=city_id,
                # Sample search string: used-cars+in+delhi
                search_string=f"used-cars+in+{city}"
            )))
        
        all_base_case_results = await asyncio.gather(*base_case_tasks)
        
        # Flatten the list of lists into a single list of cars
        all_base_case_cars = [car for cars_list in all_base_case_results for car in cars_list]
        
        print(f"Saving all {len(all_base_case_cars)} base case cars to a file...")
        # Save the base case cars to a file
        df_base = pd.DataFrame(all_base_case_cars)
        # Make the file name have the current date and time
        filename = f"cardekho_base_case_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df_base.to_csv(f"../data/{filename}", index=False)
        print(f"Saved all the base cars to a file: {filename}")
    
    async def get_and_save_rest_cars():
        tasks = []
        for city in non_base_case_cities:
            city_id = params.common_city_ids.get(city)
            city = city.replace(" ", "-").lower()
            for brand in non_base_case_brands:
                brand = brand.replace(" ", "-").lower()
                for fuel_type in non_base_case_fuel_types:
                    fuel_type = fuel_type.replace(" ", "-").lower()
                    search_string = f"used-cars+in+{city}+{brand}+{fuel_type}"
                    # all_cars.extend(await get_all_cars(city_id, search_string))
                    tasks.append(asyncio.create_task(get_all_cars(city_id, search_string)))
        
        all_results = await asyncio.gather(*tasks)
        # Flatten the list of lists into a single list of cars
        all_non_base_case_cars = [car for cars_list in all_results for car in cars_list]
        
        df_non_base = pd.DataFrame(all_non_base_case_cars)
        # Make the file name have the current date and time
        filename = f"cardekho_non_base_case_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df_non_base.to_csv(f"../data/{filename}", index=False)
    
    print("Please enter one of the following options:")
    print("1. Get and save base case cars")
    print("2. Get and save non base case cars")
    
    option = input("Enter your option: ")
    if option == "1":
        await get_and_save_base_case_cars()
    elif option == "2":
        await get_and_save_rest_cars()
    else:
        print("Invalid option. Exiting...")
        exit(1)
       

if __name__ == "__main__":
    asyncio.run(main())
