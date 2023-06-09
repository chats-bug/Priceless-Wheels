from __future__ import annotations

import datetime
import os

import pandas as pd  # data processing, CSV file I/O

from src.utils.constants import INDEX, TARGET, CLEAN_DIR_PATH, CLEAN_FILE_BEGIN, RAW_DIR_PATH, RAW_FILE_BEGIN
from src.utils.data_loader import DataLoader
from src.utils.utils import Utility as cutil


class Cleaning:
    def __init__(
            self,
            filepath: str,
            df: pd.DataFrame = None,
            index: str | None = None
    ):
        assert filepath is not None or df is not None, "Either filepath or df must be provided"
        assert filepath is None or isinstance(filepath, str), "filepath must be a string"
        assert df is None or isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert index is None or isinstance(index, str), "index must be a string"
        
        if df is None:
            df = pd.read_csv(filepath, index_col=index)
        self.df = df
        self.index = index
        return
    
    def drop_columns(self, columns: list[str]):
        self.df.drop(columns, axis=1, inplace=True)
        return
    
    def drop_columns_except(self, columns: list[str]):
        self.df.drop(self.df.columns.difference(columns), axis=1, inplace=True)
        return
    
    def drop_rows(self, rows: list[int]):
        self.df.drop(rows, inplace=True)
        return
    
    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return
    
    def str_columns_to_lower(self, columns: list[str]):
        for col in columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        return
    
    def handle_value_configuration(self):
        # Replace all the variants of `dohc` to simply `dohc`
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'dohc with vis', 'dohc')
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'dohc with vgt', 'dohc')
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            '16-valve dohc layout', 'dohc')
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'dohc with tis', 'dohc')
        
        # Replace `undefined`, `mpfi`, `vtec` with `NaN`
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'undefined', 'nan')
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'mpfi', 'nan')
        self.df['Value Configuration'] = self.df['Value Configuration'].str.replace(
            'vtec', 'nan')
        return
    
    def handle_gear_box(self):
        gear_box_mapping = {}
        gear_box_mapping['1 speed'] = [
            'single speed',
            'single speed automatic',
            'single speed reduction gear',
            'single-speed transmission',
        ]
        gear_box_mapping['4 speed'] = [
            '4 speed',
            '4-speed',
        ]
        gear_box_mapping['5 speed'] = [
            '5',
            '5 - speed',
            '5 gears',
            '5 manual',
            '5 speed',
            '5 speed at+ paddle shifters',
            '5 speed cvt',
            '5 speed forward, 1 reverse',
            '5 speed manual',
            '5 speed manual transmission',
            '5 speed+1(r)',
            '5 speed,5 forward, 1 reverse',
            '5-speed',
            '5-speed`',
            'five speed',
            'five speed manual',
            'five speed manual transmission',
            'five speed manual transmission gearbox',
        ]
        gear_box_mapping['6 speed'] = [
            '6',
            '6 speed',
            '6 speed at',
            '6 speed automatic',
            '6 speed geartronic',
            '6 speed imt',
            '6 speed ivt',
            '6 speed mt',
            '6 speed with sequential shift',
            '6-speed',
            '6-speed at',
            '6-speed automatic',
            '6-speed autoshift',
            '6-speed cvt',
            '6-speed dct',
            '6-speed imt',
            '6-speed ivt',
            '6-speed`',
            'six speed  gearbox',
            'six speed automatic gearbox',
            'six speed automatic transmission',
            'six speed geartronic, six speed automati',
            'six speed manual',
            'six speed manual transmission',
            'six speed manual with paddle shifter',
        ]
        gear_box_mapping['7 speed'] = [
            '7 speed',
            '7 speed 7g-dct',
            '7 speed 9g-tronic automatic',
            '7 speed cvt',
            '7 speed dct',
            '7 speed dsg',
            '7 speed dual clutch transmission',
            '7 speed s tronic',
            '7-speed',
            '7-speed dct',
            '7-speed dsg',
            '7-speed pdk',
            '7-speed s tronic',
            '7-speed s-tronic',
            '7-speed steptronic',
            '7-speed stronic',
            '7g dct 7-speed dual clutch transmission',
            '7g-dct',
            '7g-tronic automatic transmission',
            'amg 7-speed dct',
            'mercedes benz 7 speed automatic',
        ]
        gear_box_mapping['8 speed'] = [
            '8',
            '8 speed',
            '8 speed cvt',
            '8 speed multitronic',
            '8 speed sport',
            '8 speed tip tronic s',
            '8 speed tiptronic',
            '8-speed',
            '8-speed automatic',
            '8-speed automatic transmission',
            '8-speed dct',
            '8-speed steptronic',
            '8-speed steptronic sport automatic transmission',
            '8-speed tiptronic',
            '8speed',
            'amg speedshift dct 8g',
        ]
        gear_box_mapping['9 speed'] = [
            '9 -speed',
            '9 speed',
            '9 speed tronic',
            '9-speed',
            '9-speed automatic',
            '9g tronic',
            '9g-tronic',
            '9g-tronic automatic',
            'amg speedshift 9g tct automatic',
        ]
        gear_box_mapping['10 speed'] = [
            '10 speed',
        ]
        gear_box_mapping['cvt'] = [
            'cvt',
            'e-cvt',
            'ecvt',
        ]
        gear_box_mapping['direct drive'] = [
            'direct drive',
        ]
        gear_box_mapping['fully automatic'] = [
            'automatic transmission',
            'fully automatic',
        ]
        gear_box_mapping['nan'] = [
            'nan',
            'ags',
            'imt',
            'ivt',
        ]
        
        mapping_dict = {v: k for k, lst in gear_box_mapping.items()
                        for v in lst}
        self.df['Gear Box'] = self.df['Gear Box'].replace(mapping_dict)
        return
    
    def handle_drive_type(self):
        drive_type_mapping = {}
        drive_type_mapping['fwd'] = ['fwd', 'front wheel drive']
        drive_type_mapping['2wd'] = [
            '2wd', 'two wheel drive', '2 wd', 'two whhel drive']
        drive_type_mapping['rwd'] = [
            'rwd', 'rear wheel drive with esp', 'rear-wheel drive with esp', 'rwd(with mtt)']
        drive_type_mapping['awd'] = ['awd', 'all wheel drive',
                                     'all-wheel drive with electronic traction', 'permanent all-wheel drive quattro']
        drive_type_mapping['4wd'] = ['4wd', '4 wd', '4x4', 'four whell drive']
        drive_type_mapping['nan'] = ['nan', '3']
        
        mapping_dict = {v: k for k, lst in drive_type_mapping.items()
                        for v in lst}
        self.df['Drive Type'] = self.df['Drive Type'].replace(mapping_dict)
        return
    
    def handle_steering_type(self):
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'electrical', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'electric', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'electronic', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'epas', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'mt', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'motor', 'power')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'hydraulic', 'manual')
        self.df['Steering Type'] = self.df['Steering Type'].str.replace(
            'hydraulic', 'manual')
        return
    
    def handle_brake_type(self):
        brake_type_mapping = {}
        brake_type_mapping['disc'] = [
            'disc',
            '260mm discs',
            'disc brakes',
            'disc, 236 mm',
            'discs',
            'disk',
            'multilateral disc',
            'solid disc',
            'electric parking brake',
            'abs',
        ]
        brake_type_mapping['ventilated disc'] = [
            '264mm ventilated discs',
            'booster assisted ventilated disc',
            'caliper ventilated disc',
            'disc brakes with inner cooling',
            'disc,internally ventilated',
            'vantilated disc',
            'ventilated & grooved steel discs',
            'ventilated disc',
            'ventilated disc with twin pot caliper',
            'ventilated discs',
            'ventilated disk',
            'ventillated disc',
            'ventillated discs',
            'ventlated disc',
            'ventilated drum in discs',
            'ventialte disc',
            'ventialted disc',
        ]
        brake_type_mapping['carbon ceramic'] = [
            'carbon ceramic brakes',
            'carbon ceramic brakes.',
        ]
        brake_type_mapping['disc & drum'] = [
            'disc & drum',
            '228.6 mm dia, drums on rear wheels',
            '262mm disc & drum combination',
            'drum in disc',
            'drum in discs',
        ]
        brake_type_mapping['drum'] = [
            'drum',
            '203mm drums',
            'drum`',
            'drums',
            'drums 180 mm',
            'booster assisted drum',
            'drum brakes',
            'leading & trailing drum',
            'leading-trailing drum',
            'self adjusting drum',
            'self adjusting drums',
            'self-adjusting drum',
            'single piston sliding fist',
            'ventilated drum',
            'tandem master cylinder with servo assist',
        
        ]
        brake_type_mapping['caliper'] = [
            'six piston claipers',
            'twin piston sliding fist caliper',
            'vacuum assisted hydraulic dual circuit w',
            'four piston calipers',
            'disc & caliper type',
        ]
        
        mapping_dict = {v: k for k, lst in brake_type_mapping.items()
                        for v in lst}
        self.df['Front Brake Type'] = self.df['Front Brake Type'].replace(
            mapping_dict)
        self.df['Rear Brake Type'] = self.df['Rear Brake Type'].replace(
            mapping_dict)
        return
    
    def handle_tyre_type(self):
        tyre_type_mapping = {}
        tyre_type_mapping['tubeless'] = [
            'tubeless tyres',
            'tubeless',
            'tubeless tyres mud terrain',
            'tubeless tyre',
        ]
        tyre_type_mapping['tubeless radial'] = [
            'tubeless, radial',
            'tubeless,radial',
            'tubeless tyres, radial',
            'tubeless radial tyres',
            'radial, tubeless',
            'radial',
            'tubless, radial',
            'radial tubeless',
            'tubeless radial',
            'tubeless,radials',
            'tubeless radials',
            'radial,tubeless',
            'tubeless radial tyre',
            'radial, tubless',
            'tubless radial tyrees',
            'tubeless , radial',
            'tubeless, radials',
            'radial tyres',
        ]
        tyre_type_mapping['runflat'] = [
            'runflat tyres',
            'runflat',
            'tubeless,runflat',
            'run-flat',
            'runflat tyre',
            'tubeless, runflat',
            'tubeless. runflat',
            'tubeless.runflat',
            'tubeless radial tyrees',
        ]
        tyre_type_mapping['tube'] = [
            'radial with tube',
        ]
        
        mapping_dict = {v: k for k, lst in tyre_type_mapping.items()
                        for v in lst}
        self.df['Tyre Type'] = self.df['Tyre Type'].replace(mapping_dict)
        return
    
    def handle_fuel_injection(self):
        fuel_injection_mapping = {
            "Gasoline Port Injection": [
                "intelligent-gas port injection",
                "i-gpi",
                "dohc",
                "pfi"
            ],
            "Multi-Point Fuel Injection": [
                "mpfi",
                "multi-point injection",
                "mpfi+lpg",
                "mpfi+cng",
                "multipoint injection",
                "smpi",
                "mpi",
                "multi point fuel injection",
                "dpfi",
                "mfi",
                "multi point injection",
                "msds",
                "cng"
            ],
            "Electronic Fuel Injection": [
                "efi(electronic fuel injection)",
                "efi",
                "efi (electronic fuel injection)",
                "efic",
                "electronic fuel injection",
                "electronically controlled injection",
                "electronic injection system",
                "sefi",
                "egis",
                "efi (electronic fuel injection",
                "efi",
                "efi -electronic fuel injection",
            ],
            "Direct Injection": [
                "direct injection",
                "direct injectio",
                "direct fuel injection",
                "direct engine",
            ],
            "Common Rail Injection": [
                "crdi",
                "common rail",
                "common rail injection",
                "common rail direct injection",
                "common rail direct injection (dci)",
                "common-rail type",
                "advanced common rail",
                "common rail system",
                "common rail diesel",
                "pgm-fi (programmed fuel injection)",
                "pgm-fi (programmed fuel inje",
                "pgm - fi",
                "pgm-fi",
                "pgm-fi (programmed fuel inject",
                "direct injection common rail",
                "cdi"
            ],
            "Distributor-Type Fuel Injection": [
                "dedst",
                "distribution type fuel injection",
                "distributor-type diesel fuel injection",
            ],
            "Indirect Injection": [
                "indirect injection",
                "idi"
            ],
            "Gasoline Direct Injection": [
                "gdi",
                "gasoline direct injection",
                "tfsi",
                "tsi",
                "tgdi"
            ],
            "Turbo Intercooled Diesel": [
                "tcdi",
                "turbo intercooled diesel",
                "tdci"
            ],
            "Intake Port Injection": [
                "intake port(multi-point)"
            ],
            "Diesel Direct Injection": [
                "ddi",
                "ddis"
            ],
            "Variable Valve Timing Injection": [
                "dual vvt-i",
                "vvt-ie",
                "ti-vct"
            ],
            "Three-Phase AC Induction Motors": [
                "3 phase ac induction motors"
            ],
            "Electric": [
                "electric",
                "isg"
            ],
        }
        
        mapping_dict = {v: k for k, lst in fuel_injection_mapping.items()
                        for v in lst}
        self.df['Fuel Suppy System'] = self.df['Fuel Suppy System'].replace(
            mapping_dict)
        return
    
    def handle_pu(self):
        self.df['pu'] = self.df['pu'].str.replace(',', '').astype(float)
    
    def handle_max_power(self):
        self.df['Max Power Delivered'] = self.df['Max Power'].str.split(
            '@').str[0].apply(cutil.get_begin_float).astype(float)
        self.df['Max Power At'] = self.df['Max Power'].str.split(
            '@').str[1].apply(cutil.get_begin_float).astype(float)
        self.df.drop(columns=['Max Power'], inplace=True, axis=1)
        return
    
    def handle_max_torque(self):
        def get_rpm_average(x):
            x = str(x)
            if '-' in x:
                p1 = cutil.get_begin_float(x.split('-')[0])
                p2 = cutil.get_begin_float(x.split('-')[1])
                if p1 is None:
                    return p2
                if p2 is None:
                    return p1
                
                return (p1 + p2) / 2
            else:
                return cutil.get_begin_float(x)
        
        self.df['Max Torque Delivered'] = self.df['Max Torque'].str.split(
            '@').str[0].apply(cutil.get_begin_float).astype(float)
        self.df['Max Torque At'] = self.df['Max Torque'].str.split(
            '@').str[1].apply(get_rpm_average).astype(float)
        self.df.drop(columns=['Max Torque'], inplace=True, axis=1)
        return
    
    def handle_borex_stroke(self):
        self.df['Bore'] = self.df['BoreX Stroke'].str.split(
            'x').str[0].apply(cutil.get_begin_float).astype(float)
        self.df['Stroke'] = self.df['BoreX Stroke'].str.split(
            'x').str[1].apply(cutil.get_begin_float).astype(float)
        self.df.drop(columns=['BoreX Stroke'], inplace=True, axis=1)
        return
    
    def handle_turbo_charger(self):
        self.df['Turbo Charger'] = self.df['Turbo Charger'].replace(
            'yes', True)
        self.df['Turbo Charger'] = self.df['Turbo Charger'].replace(
            'no', False).astype(bool)
        return
    
    def handle_super_charger(self):
        self.df['Super Charger'] = self.df['Turbo Charger'].replace(
            'yes', True)
        self.df['Super Charger'] = self.df['Turbo Charger'].replace(
            'no', False).astype(bool)
        return
    
    def handle_length(self):
        self.df['Length'] = self.df['Length'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_width(self):
        self.df['Width'] = self.df['Width'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_height(self):
        self.df['Height'] = self.df['Height'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_wheel_base(self):
        self.df['Wheel Base'] = self.df['Wheel Base'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_front_tread(self):
        self.df['Front Tread'] = self.df['Front Tread'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_rear_tread(self):
        self.df['Rear Tread'] = self.df['Rear Tread'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_kerb_weight(self):
        self.df['Kerb Weight'] = self.df['Kerb Weight'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_gross_weight(self):
        self.df['Gross Weight'] = self.df['Gross Weight'].apply(
            cutil.get_begin_number).astype(float)
        return
    
    def handle_turning_radius(self):
        self.df['Turning Radius'] = self.df['Turning Radius'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_top_speed(self):
        self.df['Top Speed'] = self.df['Top Speed'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_acceleration(self):
        self.df['Acceleration'] = self.df['Acceleration'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_cargo_volumn(self):
        self.df['Cargo Volumn'] = self.df['Cargo Volumn'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_ground_clearance_unladen(self):
        self.df['Ground Clearance Unladen'] = self.df['Ground Clearance Unladen'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_compression_ratio(self):
        self.df['Compression Ratio'] = self.df['Compression Ratio'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_alloy_wheel_size(self):
        self.df['Alloy Wheel Size'] = self.df['Alloy Wheel Size'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def handle_km(self):
        self.df['km'] = self.df['km'].apply(
            cutil.get_begin_float).astype(float)
        
        # Convert to integer
        self.df['km'] = self.df['km'].astype(int)
        return
    
    def handle_mileage_new(self):
        """
        Convert mileage_new to float
        """
        self.df['mileage_new'] = self.df['mileage_new'].apply(
            cutil.get_begin_float).astype(float)
        return
    
    def rename_columns(self, renames: dict = None):
        if renames is None:
            renames = {
                'tt': 'transmission',
                'bt': 'body',
                'ft': 'fuel',
                'variantName': 'variant',
                'km': 'km_driven',
                'pu': 'listed_price',
                'Values per Cylinder': 'Valves per Cylinder',
                'Value Configuration': 'Valve Configuration',
                'No Door Numbers': 'Doors',
                'Seating Capacity': 'Seats',
                'Cargo Volumn': 'Cargo Volume',
                'city_x': 'City',
            }
        
        self.df.rename(columns=renames, inplace=True)
        return
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the dataframe
        """
        return self.df
    
    def save_data(self, file_path) -> bool:
        self.df.to_csv(file_path, index=True)
        return True


def run_cleaning_process(
        filepath: str = None,
        df: pd.DataFrame = None,
        index: str = None,
        save_to_file: bool = True,
) -> str | pd.DataFrame:
    # The filepath was set as following:
    # cardekho_cars_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv
    # Get the latest file from the data folder
    if df is None:
        if filepath is None:
            dl = DataLoader(dir_path=RAW_DIR_PATH)
            filename = dl.get_latest_file(begins_with=RAW_FILE_BEGIN)
            filepath = os.path.join(RAW_DIR_PATH, filename)
        
        df = pd.read_csv(filepath)
    index = INDEX
    
    # Create a new instance of the Cleaning class
    cdc = Cleaning(filepath=filepath, df=df, index=index)
    
    columns_to_keep = [
        "loc",
        "myear",
        "bt",
        "tt",
        "ft",
        "km",
        "ip",
        "images",
        "imgCount",
        "threesixty",
        "dvn",
        "oem",
        "model",
        "variantName",
        "city_x",
        "pu",
        "discountValue",
        "utype",
        "carType",
        "top_features",
        "comfort_features",
        "interior_features",
        "exterior_features",
        "safety_features",
        "Color",
        "Engine Type",
        "Displacement",
        "mileage_new",
        "Max Power",
        "Max Torque",
        "No of Cylinder",
        "Values per Cylinder",
        "Value Configuration",
        "BoreX Stroke",
        "Turbo Charger",
        "Super Charger",
        "Length",
        "Width",
        "Height",
        "Wheel Base",
        "Front Tread",
        "Rear Tread",
        "Kerb Weight",
        "Gross Weight",
        "Gear Box",
        "Drive Type",
        "Seating Capacity",
        "Steering Type",
        "Turning Radius",
        "Front Brake Type",
        "Rear Brake Type",
        "Top Speed",
        "Acceleration",
        "Tyre Type",
        "No Door Numbers",
        "Cargo Volumn",
        "model_type_new",
        "state",
        "owner_type",
        "exterior_color",
        "Fuel Suppy System",
        "Compression Ratio",
        "Alloy Wheel Size",
        "Ground Clearance Unladen",
    ]
    
    # Drop the columns that are not needed
    cdc.drop_columns_except(columns_to_keep)
    
    # Drop duplicate rows
    cdc.drop_duplicates()
    
    columns_str_to_lower = [
        'loc',
        'bt',
        'ft',
        'tt',
        'images',
        'dvn',
        'oem',
        'model',
        'variantName',
        'city_x',
        'utype',
        'carType',
        'top_features',
        'comfort_features',
        'interior_features',
        'exterior_features',
        'safety_features',
        'Color',
        'Engine Type',
        "mileage_new",  # 25.4 kmpl -> 25.4
        'Max Power',  # Max Power : 33.54bhp@4000 rpm -> 2 columns 33.54, 4000
        'Max Torque',  # Max Torque : 40.2Nm@3500 rpm -> 2 columns 40.2, 3500
        # No of Cylinder : float -> int,
        # Values per Cylinder : float -> int, !! Fix the name -> Valve per Cylinder
        'Value Configuration',  # !! Fix the name -> Valve Configuration
        'BoreX Stroke',  # BoreX Stroke : 69 x 72 mm -> Bore: 69, Stroke: 72
        'Turbo Charger',  # Convert to boolean
        'Super Charger',  # Convert to boolean
        'Length',  # Length : 3599mm -> 3599
        'Width',  # Width : 1495mm -> 1495
        'Height',  # Height : 1700mm -> 1700
        'Wheel Base',  # Wheel Base : 2400mm -> 2400
        'Front Tread',  # Front Tread : 1295mm -> 1295
        'Rear Tread',  # Rear Tread : 1295mm -> 1295
        'Kerb Weight',  # Kerb Weight : 960kg -> 960
        'Gross Weight',  # Gross Weight : 1350kg -> 1350
        'Gear Box',  # might need some additional cleaning
        'Drive Type',  # might need some additional cleaning
        # Seating Capacity : float -> int
        'Steering Type',  # might need some additional cleaning
        'Turning Radius',  # Turning Radius : 4.6 metres -> 4.6
        'Front Brake Type',  # might need some additional cleaning
        'Rear Brake Type',  # might need some additional cleaning
        'Top Speed',  # Top Speed : 137 kmph -> 137
        'Acceleration',  # Acceleration : 13.5 seconds -> 13.5
        'Tyre Type',  # might need some additional cleaning
        # No Door Numbers : float -> int !! fix name
        'Cargo Volumn',  # 'Cargo Volumn' : 300 litres -> 300, !! fix name
        'model_type_new',  # !! fix name
        'state',
        'exterior_color',
        'owner_type',  # might need some additional cleaning
        'Fuel Suppy System',  # might need some additional cleaning
        # Compression Ratio : 10.0:1 -> 10.0
        'Alloy Wheel Size',  # Alloy Wheel Size : convert to float
        'Ground Clearance Unladen',  # Ground Clearance Unladen : 170mm -> 170
    ]
    # Standardize the string columns
    cdc.str_columns_to_lower(columns_str_to_lower)
    
    # Get all the functions that start with 'handle_'
    # and call them
    handler_functions = [func for func in dir(cdc) if callable(
        getattr(cdc, func)) and func.startswith('handle_')]
    for func in handler_functions:
        getattr(cdc, func)()
    
    # Rename the columns
    cdc.rename_columns()
    if not save_to_file:
        return cdc.get_data()
    
    save_filename = f"{CLEAN_FILE_BEGIN}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    save_filepath = os.path.join(CLEAN_DIR_PATH, save_filename)
    # Save the cleaned data
    cdc.save_data(save_filepath)
    
    return save_filepath


def main():
    print('Cleaning the data...')
    df = pd.read_csv('../../data/raw/cardekho_cars_2023_03_19_16_44_14.csv', index_col=INDEX)
    file_path = run_cleaning_process(df=df, save_to_file=True)
    print(f'Cleaned data saved to {file_path}')


if __name__ == '__main__':
    main()
