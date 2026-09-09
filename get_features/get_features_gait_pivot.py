## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_gait_pivot():
    print('Running gait_pivot...')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            gait_pivot_features = {}  
            trial_names = ["gait-pivot"]
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
        
            # EXTRACT COORDINATE VALUES, SPEEDS, and MARKERS
            coordinate_values = kinematics[trial_name].get_coordinate_values(in_degrees=True)
            coordinate_speeds = kinematics[trial_name].get_coordinate_speeds(in_degrees = True)
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5)
            wrist_peak_idx = wrist_peak_idx[0]
            sample_freq = round(1/marker_dict['time'][1])
            go_idx = wrist_peak_idx + (2*sample_freq)
        
            # LOW PASS FILTER #########################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################          
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
        
            pelvis_rot = coordinate_values['pelvis_rotation']
            pelvis_rot_speed = coordinate_speeds['pelvis_rotation']
            pelvis_rot_speed_filt = filtfilt(b,a,pelvis_rot_speed)
            
            i = go_idx
            while pelvis_rot[i] < 100 and pelvis_rot[i] > -100: i+=1
            temp = i
            
            i = temp
            while pelvis_rot[i] > 20 or pelvis_rot[i] < -20: i-=1
            turn_start_20_idx = i
            
            i = temp
            while pelvis_rot[i] > -160 and pelvis_rot[i] < 160: i+=1
            turn_end_160_idx = i
            
            while pelvis_rot[i] > -170 and pelvis_rot[i] < 170: i+=1
            turn_end_170_idx = i
            
            i = turn_start_20_idx
            while COM_speeds_x[i] > 0.1: i-=1
            
            i = turn_end_170_idx+30
            while COM_speeds_x[i] > -0.2: i+=1
            gait_back_idx = i
              
            ## GET FEATURES ###############################################################
            mean_turn_speed = np.abs(np.mean(pelvis_rot_speed_filt[turn_start_20_idx:turn_end_160_idx]))

            calc_r_x = marker_dict['markers']['r_calc_study'][:,0]
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            calc_l_x = marker_dict['markers']['L_calc_study'][:,0]
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]

            calc_dist = np.sqrt((calc_r_x - calc_l_x)**2 + (calc_r_z - calc_l_z)**2)
            range_ankle_dist = np.max(calc_dist[turn_end_170_idx:gait_back_idx]) -np.min(calc_dist[turn_end_170_idx:gait_back_idx])
    
            gait_pivot_features = {"mean_turn_speed": mean_turn_speed, "range_ankle_dist": range_ankle_dist,
                                   }
            filepath = 'features/gait_pivot_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(gait_pivot_features, json_file)
    print('Finished')