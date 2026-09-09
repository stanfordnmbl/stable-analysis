## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json


def get_features_gait_fast():
    print("Running gait_fast... ")
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            gait_fast_features = {}  
            trial_names = ["gait-fast"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
        
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0] 
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
            
            # GET CENTER OF MASS VALUES, SPEEDS
            center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5)
            wrist_peak_idx = wrist_peak_idx[0]
            sample_freq = round(1/marker_dict['time'][1])
            go_idx = wrist_peak_idx + (2*sample_freq)
        
            # SEGMENTATION ###############################################################    
            COM_values_x = center_of_mass['values'][trial_name]['x']
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
        
            neutral_x = COM_values_x[go_idx]
            start_x = neutral_x + 0.5
            end_x = start_x + 2.5
            
            i = go_idx
            while COM_values_x[i] < start_x: i+=1
            start_idx = i
            while COM_values_x[i] < end_x: i+=1
            end_idx = i
              
            # GET FEATURES ###############################################################
            start_speed = np.mean(COM_speeds_x[start_idx:start_idx+30])
            end_speed = np.mean(COM_speeds_x[end_idx:end_idx+30])
            speed_change = end_speed - start_speed
            
            gait_fast_features = {"speed_change": speed_change}
            filepath = 'features/gait_fast_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
                json.dump(gait_fast_features, json_file)
    print('Finished')