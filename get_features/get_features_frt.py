## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_frt():
    print("Running frt...")
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            frt_features = {}  
            trial_names = ["frt"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
        
            # Get mass from session metadata
            import yaml
            with open(f'{data_folder}/sessionMetadata.yaml', 'r') as file:
                metadata = yaml.safe_load(file)
            height = metadata['height_m']

            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0] 
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5)
            wrist_peak_idx = wrist_peak_idx[0]
        
            ## SEGMENTATION ###############################################################           
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]
            
            toe_r_x = marker_dict['markers']['r_toe_study'][:,0]
            toe_l_x = marker_dict['markers']['L_toe_study'][:,0]
            
            lwrist_r_x = marker_dict['markers']['r_lwrist_study'][:,0]
            lwrist_r_z = marker_dict['markers']['r_lwrist_study'][:,2]
            lwrist_l_x = marker_dict['markers']['L_lwrist_study'][:,0]
            lwrist_l_z = marker_dict['markers']['L_lwrist_study'][:,2]
        
            ## GET FEATURES ###############################################################    
            max_wrist_toe_r_x = np.max(lwrist_r_x - toe_r_x)
            max_wrist_toe_l_x = np.max(lwrist_l_x - toe_l_x)       
            max_wrist_calc_r_z = np.max(np.abs(lwrist_r_z - calc_r_z))
            max_wrist_calc_l_z = np.max(np.abs(lwrist_l_z - calc_l_z))
    
            fwd_dist = np.mean((max_wrist_toe_r_x, max_wrist_toe_l_x)) /  height
            r_dist = max_wrist_calc_r_z / height
            l_dist = max_wrist_calc_l_z / height
            
            frt_features = {"fwd_dist": fwd_dist, "r_dist": r_dist, "l_dist": l_dist }
            filepath = 'features/frt_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(frt_features, json_file)
    print('Finished')