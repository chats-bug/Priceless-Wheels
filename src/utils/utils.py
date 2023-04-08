import os
import datetime


class Utility:
    """
    A utility class that contains all the utility functions.
    """
    def convert_to_number(x: str, conv: str = 'float'):
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
    
    @staticmethod
    def get_begin_number(x: str):
        return Utility.convert_to_number(x, 'int')

    @staticmethod
    def get_begin_float(x: str):
        return Utility.convert_to_number(x, 'float')
    

if __name__ == "__main__":
    print(Utility.get_begin_number('120-litres'))