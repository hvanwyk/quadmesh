'''
Created on Apr 7, 2016

@author: hans-werner
'''

def build_buoy_lookup(data_files, lookup_file):
    '''
    Build a lookup table listing all buoys and where to locate information on
    them in the datafiles.
    
    Inputs
        
        datafiles: str, list of filenames containing drifter information
        
    Output
    
        buoy_lookup_table.dat: file in drifter/data, to store lookup table
            headers: ID    file_num    first_line    last_line
    '''
    #
    # Import Data
    # 
    current_id = 'None'
    list_id = []
    list_file_num = []
    list_first_line = []
    list_last_line = []
    #
    # Loop over data files
    #
    file_num = 0
    for file_name in data_files:
        #
        # Open data file
        # 
        with open(file_name,"r") as fin:
            print('Opening read file: %s' % file_name)
            file_num += 1
            line_count = 0
            for line in fin:
                line_count += 1
                line = line.split()
                if current_id != line[0]:
                    #
                    # New buoy!
                    #
                    if line_count > 1:
                        #
                        # Record previous buoy's data
                        # 
                        list_last_line.append(line_count - 1)
                    #
                    # Update data
                    # 
                    current_id = line[0]
                    list_id.append(current_id)
                    list_first_line.append(line_count)
                    list_file_num.append(file_num)

    # 
    # Sort according to ID
    # 
    
    #
    # Open lookup file
    #
    """with open(lookup_file, "w") as fout:
        print('Opening file: %s' % lookup_file)            
        buoy_info_line = "%8s %d %-8d %-8d \n"
                     % (current_id, file_num,
                        first_line, last_line)
                        fout.write(buoy_info_line)
if __name__ == '__main__':
    data_file = ['buoydata_1_5000.dat', \
                 'buoydata_5001_10000.dat',\
                 'buoydata_10001_jun15.dat']
    lookup_file = "buoy_lookup_table.dat"
    build_buoy_lookup(data_file, lookup_file)"""
                
