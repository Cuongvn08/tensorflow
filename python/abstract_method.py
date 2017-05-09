# -*- coding: utf-8 -*-


import abc

class image_base(object):
    __metaclass__ = abc.ABCMeta        
        
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def get_width(self):
        return self.width
        
    def get_height(self):
        return self.height
    
    @abc.abstractmethod
    def get_size(self):
        return(self.width * self.height)

class image_A(image_base):
    def get_size(self):
        return(self.width + self.height)

# test
img_base = image_base(810, 1115)        
print(img_base.get_size()) # 903150 (width * height)

img_A = image_A(810, 1115)
print(img_A.get_size()) # 1925 (width + height)