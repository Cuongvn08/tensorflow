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
    
    def test(self):
        width = self.get_width()
        height = self.get_height()
        return (width * height)
   
# test    
img = image(810, 1115)
print(img.get_width())
print(img.get_height())