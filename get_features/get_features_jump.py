## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json


def get_features_jump():
    print('Running jump...')
    record_ids = ['42', '68', '140', '144']
    for i in range(4):
        record_id = record_ids[i]
        jump_features = {}  
        for trial in ["jump-r", "jump-l"]:
                trial_names = [trial]
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
                
                # GET CENTER OF MASS VALUES, SPEEDS, ACCELERATIONS
                center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
                COM_speeds_y = center_of_mass['speeds'][trial_name]['y']

                # EXTRACT MARKERS
                marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
            
                # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
                signal= marker_dict['markers']['r_lwrist_study'][:,1]
                wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
                wrist_peak_idx = wrist_peak_idx[0]
                sample_freq = round(1/marker_dict['time'][1])
            
                # CUES #####################################################################
                go_idx = wrist_peak_idx + (1*sample_freq)
            
                ## SEGMENTATION ###############################################################      
                LO = np.argmax(COM_speeds_y[go_idx:go_idx+round(25/6*sample_freq)]) + go_idx
                TD = np.argmin(COM_speeds_y[LO:LO+round(4/6*sample_freq)]) + LO
            
                time = marker_dict['time']
                if trial == "jump-r" or trial == "jump-r_yes":
                    calc_land_x = marker_dict['markers']['r_calc_study'][:,0]
                    calc_land_z = marker_dict['markers']['r_calc_study'][:,2]
                    toe_lift_y = marker_dict['markers']['L_toe_study'][:,1]
                elif trial == "jump-l":
                    calc_land_x = marker_dict['markers']['L_calc_study'][:,0]
                    calc_land_z = marker_dict['markers']['L_calc_study'][:,2]
                    toe_lift_y = marker_dict['markers']['r_toe_study'][:,1]    
            
                ## GET FEATURES ###############################################################
                x_dist = calc_land_x[TD] - calc_land_x[LO]
                z_dist = calc_land_z[TD] - calc_land_z[LO]
                jump_dist = np.sqrt(x_dist**2 + z_dist**2)/height
            
                i = TD
                hop = i
                hop_land = i-7
                while np.abs((np.sqrt(calc_land_x**2 + calc_land_z**2)[hop_land+7] - np.sqrt(calc_land_x**2 + calc_land_z**2)[hop])) < 0.01 and i < len(COM_speeds_y)-8 and hop_land<len(COM_speeds_y)-8:
                    while COM_speeds_y[i] < 0.3 and i < len(COM_speeds_y)-1: i+=1
                    hop = i
                    while COM_speeds_y[i] > 0 and i < len(COM_speeds_y)-1: i+=1
                    while COM_speeds_y[i] < 0 and COM_speeds_y[i] > -0.2 and i < len(COM_speeds_y)-1: i+=1
                    hop_land = i
                    if hop_land + 7 >len(COM_speeds_y)-1: hop_land -=8
                if COM_speeds_y[i] > -0.2: 
                    hop = len(COM_speeds_y)-8
                    hop_land = len(COM_speeds_y)-8
                
                i = TD + 1
                while (toe_lift_y[i] > toe_lift_y[go_idx]+0.01) and i < len(toe_lift_y)-1: i+=1
                lift_down = i
                time_land_touch_raw = time[lift_down] - time[TD]
                time_land_touch = np.min([time_land_touch_raw, 3])
                
                time_land_hop_raw = time[hop] - time[TD]
                time_land_hop = np.min([time_land_hop_raw, 3, time_land_touch_raw])
    
                if trial == "jump-r" or trial == "jump-r_yes": side = 'r'
                elif trial == "jump-l": side = 'l'
                jump_features = {f"{side}_time_land_touch": time_land_touch, f"{side}_time_land_hop": time_land_hop, f"{side}_x_dist": x_dist, f"{side}_z_dist": z_dist, f"{side}_jump_dist": jump_dist }
            
        filepath = 'features/jump_features_' + record_id + '.json'
        
        # Initialize a dictionary to hold existing data if the file does not exist
        if not os.path.exists(filepath):
            existing_data = {}
        else:
            # Read existing data
            with open(filepath, 'r') as json_file:
                existing_data = json.load(json_file)
           
        # Update the existing data with the new dictionary
        existing_data.update(jump_features)
            
        # Write the updated data back to the JSON file
        with open(filepath, 'w') as json_file:
            json.dump(existing_data, json_file) 
    print('Finished')