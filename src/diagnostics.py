'''
Created on Jun 10, 2019

@author: hans-werner
'''
import time 


class Verbose(object):
    """
    Class for producing comments and timing snippets of code
    """


    def __init__(self):
        """
        Constructor
        """
        self.__tic = None
        self.__toc = None
    
        
    def comment(self, string):
        """
        Print a comment
        """
        print(string)
        
            
    def tic(self, string=None):
        """
        Start timer
        """
        if string is not None:
            print(string,end="")
        self.__tic = time.time()


    def toc(self):
        """
        Print the time elapsed since tic
        """
        assert self.__tic is not None, 'Use "tic" to start timer.'
        toc = time.time()
        print(' (time elapsed %.4f sec)'%(toc-self.__tic))
        self.__tic = None
    
    
 
    '''
    def deep_getsizeof(self, o, ids): 
        """
        Find the memory footprint of a Python object
    
        This is a recursive function that drills down a Python object graph
        like a dictionary holding nested dictionaries with lists of lists
        and tuples and sets.
        
        The sys.getsizeof function does a shallow size of only. It counts each
        object inside a container as pointer only regardless of how big it
        really is.
        
        :param o: the object
        :param ids:
        :return:
        """
        d = deep_getsizeof
        if id(o) in ids:
            return 0
        
        r = getsizeof(o)
        ids.add(id(o))
        
        if isinstance(o, str) or isinstance(0, unicode):
            return r
        
        if isinstance(o, Mapping):
            return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
        
        if isinstance(o, Container):
            return r + sum(d(x, ids) for x in o)
    '''