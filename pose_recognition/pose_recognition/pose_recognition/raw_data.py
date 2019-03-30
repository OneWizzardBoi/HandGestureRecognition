import numpy as np
import json

from os import listdir
from os.path import isfile, isdir, join


# Gets all the files in the specified directory and in the sub directories
# Inputs : 
#   target_dir : path to the target directory
#   filter_funct : filter to be applied to collected files
# Returns : list of all the files paths for the files located in the folder/sub-folders
def fetch_training_files(target_dir, filter_funct=None):

    f_paths = []
    if not isdir(target_dir) : return f_paths

    # collecting all .json file paths in the current directory
    f_paths = [target_dir + "/" + f for f in listdir(target_dir) 
                        if isfile(join(target_dir, f)) and f.endswith('.json')]
    
    # aplying a filter to the collected file paths
    if filter_funct is not None:
        f_paths = list(filter(filter_funct, f_paths))

    # collecting all sub directory paths in current directory
    sub_directories = [target_dir + "/" + element for element in listdir(target_dir) 
                      if isdir(join(target_dir, element))]
    
    # scanning all the sub directories for files
    for sub_dir in sub_directories:
        f_paths += fetch_training_files(sub_dir, filter_funct)

    return f_paths 


# defining a filter to exclude rest acquisitions
def filter_out_rest(file_path):
    filter_vocab = ["rest"]
    for word in filter_vocab:
        if word in file_path : return False
    return True

# defining a filter to only include rest acquisitions
def filter_in_rest(file_path):
    filter_vocab = ["rest"]
    for word in filter_vocab:
        if word in file_path : return True
    return False


# Transforms a movement sampled using the (1,1,1) method to data which can be used for training
# Cuts the rest period before a movement transition and the hold period after
# Inputs : 
#   f_name : name of the .json file containing the acquisition data
#   mov_len : length of the period in acquisitions (200 acquisitions = 1 second)
#   window_offset : once the transition window is found, its move by (window_offset)
# Ouputs : none, the function will modify the specified file 
def cutout_transition(f_name, mov_len=200, window_offset=0):

    # defining basic constants
    n_channels = 8
    avg_group_size = 5

    # loading data with shape : (8, n_acquisitions)
    json_data = json.loads(open(f_name, 'r').read())
    emg_data = np.swapaxes(np.array(json_data["emg"]["data"], dtype=np.dtype(np.int32)), 0,1)
    n_points = emg_data.shape[1]

    # return if the file does not contain enough data 
    if n_points < mov_len : 
        print("Error, file : ", f_name, "does not contain enough data for processing.")
        return

    # sampling avgs for all channels of the acquisition
    n_avgs = n_points // avg_group_size
    channel_avgs = np.zeros((n_channels, n_avgs))
    for c_i in range(n_channels):
        for i in range(n_avgs):            
            channel_avgs[c_i][i] = np.mean(np.absolute(emg_data[c_i][i * avg_group_size: (i+1) * avg_group_size]))

    # recording the max amplitude for each channel
    max_amplitudes = 8 * [0]
    for c_i in range(n_channels):
        max_amplitudes[c_i] = np.amax(channel_avgs[c_i][:]) - np.amin(channel_avgs[c_i][:])
    
    # channel with the largest amplitude gets selected
    selected_i = max_amplitudes.index(max(max_amplitudes))
    data_avgs = channel_avgs[selected_i]

    # marking the start of transition at the first avg to reach 25% of max avg
    movement_start_i = 0
    treshold_avg = 0.25 * max(data_avgs)
    for i in range(len(data_avgs)): 
        if(data_avgs[i] >= treshold_avg): 
            movement_start_i = i * avg_group_size - avg_group_size
            break

    # applying the offset to the transition window
    movement_start_i += window_offset
    if(movement_start_i < 0): movement_start_i = 0

    # cutting the acquisition data
    emg_data = emg_data[ : , movement_start_i : (movement_start_i + mov_len)]

    # writting the transformed data back into the file
    emg_data = np.swapaxes(emg_data, 0, 1)
    json_data["emg"]["data"] = emg_data.tolist()
    open(f_name, 'w').write(json.dumps(json_data))


# Cuts a window of data of the specified length at the specified index
# Inputs : 
#   f_name : name of the .json file containing the acquisition data
#   start_i : the data index at which the window begins
#   mov_len : length of the period in acquisitions (200 acquisitions = 1 second)
# Ouputs : none, the function will modify the specified file 
def cutout_data_window(f_name, start_i=0, mov_len=200):

    # loading data with shape : (8, n_acquisitions)
    json_data = json.loads(open(f_name, 'r').read())
    emg_data = np.swapaxes(np.array(json_data["emg"]["data"], dtype=np.dtype(np.int32)), 0,1)

    # return if the file does not contain enough data 
    n_points = emg_data.shape[1]
    if n_points < mov_len + start_i : 
        print("Error, file : ", f_name, "does not contain enough data for processing.")
        return

    # cutting the acquisition data
    emg_data = emg_data[ : , start_i : (start_i + mov_len)]

    # writting the transformed data back into the file
    emg_data = np.swapaxes(emg_data, 0, 1)
    json_data["emg"]["data"] = emg_data.tolist()
    open(f_name, 'w').write(json.dumps(json_data))


# Transforms all data in specified directory in to training ready data
# For each movement acquisition : Cuts the rest period before a movement transition and the hold period after
# For each rest acquisition : Cuts out window of specified dimensions
# Inputs : 
#   target_dirs : directories to collect 
# Outputs : none, the function will modify the specified directory 
def prepare_training_data(target_dir):

    # processing finger movement acquisitions
    training_files = fetch_training_files(target_dir, filter_out_rest)
    for t_file in training_files:
        cutout_transition(t_file, window_offset=50)

    # processing rest acquisitions
    training_files = fetch_training_files(target_dir, filter_in_rest)
    for t_file in training_files:
        cutout_data_window(t_file)