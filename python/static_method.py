# -*- coding: utf-8 -*-

class image(object):
    # Self is used in instance methods.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def get_width(self):
        return self.width
        
    def get_height(self):
        return self.height
    
    def get_size(self):
        return (self.width * self.height)
    
    # Staticmethod: don't need to create class object, just calling the function
    # Staticmethod: can not access to class members
    # directly
    @staticmethod
    def get_size_st(width,height):
        return(width*height)
    
# test    
img = image(810, 1115)    
print(img.get_size()) # 903150

print(image.get_size_st(810, 1115)) # 903150

