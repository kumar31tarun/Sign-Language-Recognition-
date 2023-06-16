#program to check which file is curropted in what directory
import os
#PIL(Python Image Library) , imports Image module
from PIL import Image

dir_path = '/Users/tarunkumar/Desktop/project/Sign Language Recognition/data'

for filename in os.listdir(dir_path):
    try:
        with Image.open(os.path.join(dir_path, filename)) as im:
            #this line checks the file by opening it, if it is curropted it will raise an exception
            im.verify()
            #this assigns the exception raised to 'e'
    except Exception as e:
        print(f'Error opening {os.path.join(dir_path, filename)}: {e}')


