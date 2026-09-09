## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json


def get_features_5tsts(): 
    print('Running 5tsts... ')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            ftsts_features = {} 
            trial_names = ["5tsts"]
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
            COM_values_y = center_of_mass['values'][trial_name]['y']
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1] 
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5) 
            wrist_peak_idx = wrist_peak_idx[0]
            sample_freq = round(1/marker_dict['time'][1])
        
            ## SEGMENTATION ###############################################################    
            go_idx = wrist_peak_idx + (2*sample_freq)
            time = marker_dict['time']
            ASIS_r_y = marker_dict['markers']['r.ASIS_study'][:,1]
            ASIS_l_y = marker_dict['markers']['L.ASIS_study'][:,1]
            PSIS_r_y = marker_dict['markers']['r.PSIS_study'][:,1]
            PSIS_l_y = marker_dict['markers']['L.PSIS_study'][:,1]
            
            pelvis_y = np.sqrt(ASIS_r_y**2 + ASIS_l_y**2 + PSIS_r_y**2 + PSIS_l_y**2) 
        
            i = go_idx
            while pelvis_y[i] < pelvis_y[go_idx] + 0.03: i += 1
            start_idx = i
    
            i = len(pelvis_y)-1
            while pelvis_y[i] < pelvis_y[-1] + 0.03: i -=1
            end_idx = i
            
            ## GET FEATURES ###############################################################
            time = time[end_idx] - time[start_idx] 
            speed = (np.max(COM_values_y) - np.min(COM_values_y))*5/time
            
            ftsts_features = {"time": time, "speed": speed}
            filepath = 'features/5tsts_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(ftsts_features, json_file)
    print('Finished')