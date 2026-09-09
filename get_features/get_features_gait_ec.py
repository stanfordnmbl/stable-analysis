## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json

def get_features_gait_ec():
    print('Running gait_ec... ')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            gait_ec_features = {}
            trial_names = ["gait-ec"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
       
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0] 
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
            
            # GET CENTER OF MASS VALUES, SPEEDS, ACCELERATIONS
            center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idx[0]
            sample_freq = round(1/marker_dict['time'][1])
            go_idx = wrist_peak_idx + (2*sample_freq)
        
            ## SEGMENTATION ###############################################################    
            COM_values_x = center_of_mass['values'][trial_name]['x']
            COM_values_z = center_of_mass['values'][trial_name]['z']
        
            neutral_x = COM_values_x[go_idx]
            start_x = neutral_x + 0.25
            end_x = start_x + 3
            
            i = go_idx
            while COM_values_x[i] < start_x and i<len(COM_values_x)-1: i+=1
            start_idx = i
            while COM_values_x[i] < end_x and i<len(COM_values_x)-1: i+=1
            end_idx = i
              
            ## GET FEATURES ###############################################################
            ml_dev = COM_values_z[end_idx] - COM_values_z[start_idx]
            ml_dev_abs = np.abs(ml_dev)
            
            gait_ec_features = {"ml_dev": ml_dev, "ml_dev_abs": ml_dev_abs}        
            filepath = 'features/gait_ec_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(gait_ec_features, json_file)
    print('Finished')