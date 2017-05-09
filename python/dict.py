# -*- coding: utf-8 -*-

image_dict = {}

def get_width():
    width = 810
    return width

def get_height():
    height = 1115
    return height

def create_dict():
    global image_dict
    image_dict = { 'width'  : get_width(),
                   'height' : get_height() }
    
# test    
create_dict()
print(image_dict['width'])
print(image_dict['height'])
    
for key in image_dict:
    print(key)