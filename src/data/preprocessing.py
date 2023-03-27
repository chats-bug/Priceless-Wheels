import datetime
import os
import numpy as np
import pandas as pd

from utils import INDEX, Utility as cutil


class Transformations:
    def myear_transformation(df: pd.DataFrame):
        """
        Remove all rows where the year is less than 2005
        """
        df = df[df['myear'] > 2005]
        return df

    def images_transformation(df: pd.DataFrame):
        """
        Drop the column 'images' from the dataframe
        """
        df.drop('images', axis=1, inplace=True)
        return

    def imgCount_transformation(df: pd.DataFrame):
        """
        Drop the column 'imgCount' from the dataframe
        """
        df.drop('imgCount', axis=1, inplace=True)
        return

    def threesixty_transformation(df: pd.DataFrame):
        """
        Drop the column 'threesixty' from the dataframe
        """
        df.drop('threesixty', axis=1, inplace=True)
        return

    def dvn_transformation(df: pd.DataFrame):
        """
        Drop the column 'dvn' from the dataframe
        """
        df.drop('dvn', axis=1, inplace=True)
        return

    def discountValue_transformation(df: pd.DataFrame):
        """
        Drop the column 'discountValue' from the dataframe
        """
        df.drop('discountValue', axis=1, inplace=True)
        return

    def carType_transformation(df: pd.DataFrame):
        """
        Drop the column 'carType' from the dataframe
        """
        df.drop('carType', axis=1, inplace=True)
        return

    def NumOfCylinder_transformation(df: pd.DataFrame):
        """
        Replace the value of No of Cylinder with null if the car is electric
        """
        df.loc[(df['fuel'] == 'Electric') & (
            df['No of Cylinder'].notnull()), 'No of Cylinder'] = np.nan
        return

    def Height_transformation(df: pd.DataFrame):
        """
        Drop the 'Height' column
        """
        df.drop('Height', axis=1, inplace=True)
        return

    def Length_transformation(df: pd.DataFrame):
        """
        Drop the 'Length' column
        """
        df.drop('Length', axis=1, inplace=True)
        return

    def RearTread_transformation(df: pd.DataFrame):
        """
        Drop the 'Rear Tread' column
        """
        df.drop('Rear Tread', axis=1, inplace=True)
        return

    def NumOfCylinder_transformation(df: pd.DataFrame):
        """
        Replace the value of No of Cylinder with null if the car is electric
        """
        df.loc[(df['fuel'] == 'Electric') & (
            df['No of Cylinder'].notnull()), 'No of Cylinder'] = np.nan
        return

    def Height_transformation(df: pd.DataFrame):
        """
        Drop the 'Height' column
        """
        df.drop('Height', axis=1, inplace=True)
        return

    def Length_transformation(df: pd.DataFrame):
        """
        Drop the 'Length' column
        """
        df.drop('Length', axis=1, inplace=True)
        return

    def RearTread_transformation(df: pd.DataFrame):
        """
        Drop the 'Rear Tread' column
        """
        df.drop('Rear Tread', axis=1, inplace=True)
        return

    def GrossWeight_transformation(df: pd.DataFrame):
        """
        Drop the 'Gross Weight' column
        """
        df.drop('Gross Weight', axis=1, inplace=True)
        return

    def Seats_transformation(df: pd.DataFrame):
        """
        Replace the Seats with null if the value is zero
        """
        df.loc[(df['Seats'] == 0), 'Seats'] = np.nan
        return

    def TurningRadius_transformation(df: pd.DataFrame):
        """
        Replace the Turning Radius with null if the value is greater and 15
        """
        df.loc[(df['Turning Radius'] > 15.0), 'Turning Radius'] = np.nan
        return

    def Doors_transformation(df: pd.DataFrame):
        """
        Replace all rows having Doors=5 with Doors=4
        """
        df.loc[(df['Doors'] == 5), 'Doors'] = 4
        return

    def model_type_new_transformation(df: pd.DataFrame):
        """
        Drop the 'model_type_new' column
        """
        df.drop('model_type_new', axis=1, inplace=True)
        return

    def exterior_color_transformation(df: pd.DataFrame):
        """
        Drop the 'exterior_color' column
        """
        df.drop('exterior_color', axis=1, inplace=True)
        return

    def GroundClearanceUnladen_transformation(df: pd.DataFrame):
        """
        Drop the 'Ground Clearance Unladen' column
        """
        df.drop('Ground Clearance Unladen', axis=1, inplace=True)
        return

    def CompressionRatio_transformation(df: pd.DataFrame):
        """
        Drop the 'Compression Ratio' column
        """
        df.drop('Compression Ratio', axis=1, inplace=True)
        return

    def AlloyWheelSize_transformation(df: pd.DataFrame):
        """
        Replace the rows having Alloy Wheel Size=7 with null
        """
        df.loc[(df['Alloy Wheel Size'] == 7), 'Alloy Wheel Size'] = np.nan
        return

    def MaxTorqueAt_transformation(df: pd.DataFrame):
        """
        Mark all the rows having Max Torque At below 1000 and above 5000 as null
        """
        df.loc[(df['Max Torque At'] < 1000) | (
            df['Max Torque At'] > 5000), 'Max Torque At'] = np.nan
        return

    def Bore_transformation(df: pd.DataFrame):
        """
        Drop the 'Bore' column
        """
        df.drop('Bore', axis=1, inplace=True)
        return

    def Stroke_transformation(df: pd.DataFrame):
        """
        Drop the 'Stroke' column
        """
        df.drop('Stroke', axis=1, inplace=True)
        return


class Missing_Values:
    def __init__(self):
        pass


def main():
    file_name = cutil.get_latest_file('cleaned_data', './data/clean/')
    filepath = os.path.join('./data/clean/', file_name)
    df = pd.read_csv(filepath, index_col=INDEX)

    # Get all the functions from the Transformations class
    suggested_transformations = [func for func in dir(Transformations) if callable(
        getattr(Transformations, func)) and not func.startswith("__")]

    # Apply all the suggested transformations
    for transformation in suggested_transformations:
        getattr(Transformations, transformation)(df)
        # print(transformation)

    # Save the transformed data
    save_filename = f"transformed_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    save_filepath = os.path.join('./data/processed/', save_filename)
    df.to_csv(save_filepath)


if __name__ == '__main__':
    main()
