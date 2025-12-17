class ParentClass(object):
    """
    """
    def __init__(self, height, length):
        self.__height = height
        self.__length = length
        
    def height(self):
        return self.__height
    
    def length(self):
        return self.__length
    
    
class ChildClass(ParentClass):
    def __init__(self, height, length, width):
        ParentClass.__init__(self, height, length)
        
        self.__width = width
        
    def width(self):
        return self.__width
    
    
    def modify_width(self, new_width):
        self.__width = new_width

height = 10
width = 4
length = 12

p = ParentClass(height, length)
print(p.height(), p.length())

c = ChildClass(height, length, width)
print(c.height(), c.length(), c.width())

c.modify_width(1) 
print(c.height(), c.length(), c.width())