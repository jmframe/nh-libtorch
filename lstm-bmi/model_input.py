import numpy as np


#--------------------------------------------------------
#  Example utility class to read data from a text file.
#--------------------------------------------------------



class input_file:
    
    #------------------------------------------------------------
    def __init__(self, filename='data.txt'):

        self.check_format()



    def open(self):
        
        try:
            f = open(self.filename, 'r')
        except IOError as err:
            errno, strerror = err.args
            print('Could not find input file named:')
            print(self.filename)
            # print "I/O error(%s): %s" % (errno, strerror)
            print()
            return
           


 
