# -*- coding: utf-8 -*-

class image(object):
    size = 1000
    
    # Self is used in instance methods.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def get_width(self):
        return self.width
        
    def get_height(self):
        return self.height
    
    # Classmethod: similar to static method but can access to all class members
    # Cls: is often used in class method
    @classmethod
    def get_size(cls):
        return (cls.size)
    
  # test  
print(image.get_size())