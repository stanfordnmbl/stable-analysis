## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
import json
from scipy.signal import find_peaks

def get_features_standing_ec(): 
    print('Running standing_ec...')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            standing_ec_features = {}  
            trial_names = ["standing-ec"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
                
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0]
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)

            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
            COM_speeds_z = center_of_mass['speeds'][trial_name]['z']
        
            # EXTRACT COORDINATE VALUES, SPEEDS, and MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
            marker_time = marker_dict['time']
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idx[0]
        
            # CUES #####################################################################
            # 0:00 lower, 0:02 hands on hips, 0:03 "go", 0:085 "neutral", 0:25 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [1, 13, 23]
            cue_idx = [(time * sample_freq) for time in cue_times] + wrist_peak_idx
        
            ## SEGMENTATION ###############################################################
            ankle_width = np.abs(marker_dict['markers']['L_calc_study'][:,2]-marker_dict['markers']['r_calc_study'][:,2])
            
            min_ankle_width = np.min(ankle_width[cue_idx[0]:cue_idx[1]])
            max_ankle_width = 0.4
            
            ankle_thresh_5 = (max_ankle_width - min_ankle_width)*0.05 + min_ankle_width
            ankle_thresh_15 = (max_ankle_width - min_ankle_width)*0.15 + min_ankle_width
            
            # EYES OPEN SEG
            i = cue_idx[0]
            while ankle_width[i] > ankle_thresh_5: i = i+1
            start_eo = i
            while ankle_width[i] < ankle_thresh_15 and i < cue_idx[1]: i = i+1
            end_eo = i
            start_5_eo = max([end_eo - 5*sample_freq, start_eo])
            
            # EYES CLOSED SEG
            start_ec = cue_idx[1]
            i = start_ec
            while ankle_width[i] < ankle_thresh_15 and i < cue_idx[2]: i = i+1
            end_ec = i
            start_5_ec = max([end_ec - 5*sample_freq, start_ec])
            
            ## GET FEATURES ###############################################################
            if start_ec == end_eo: time = marker_time[end_ec] - marker_time[start_eo]
            else: time = marker_time[end_eo] - marker_time[start_eo]
            time_eo = min(marker_time[end_eo] - marker_time[start_eo], 10)
            time_ec = marker_time[end_ec] - marker_time[start_ec]
    
            if time_eo != 0:
                mean_com_speed_eo = np.mean(np.sqrt(COM_speeds_x[start_5_eo:end_eo]**2 + COM_speeds_z[start_5_eo:end_eo]**2))
                c7_x_eo = marker_dict['markers']['C7_study'][:,0][start_5_eo:end_eo]
                c7_z_eo = marker_dict['markers']['C7_study'][:,2][start_5_eo:end_eo]
                c7_std_eo = (np.sqrt(np.std(c7_x_eo)**2 + np.std(c7_z_eo)**2)) / marker_dict['markers']['C7_study'][:,1][start_eo]
            else: 
                mean_com_speed_eo = np.nan
                c7_std_eo = np.nan
            if time_ec != 0:
                mean_com_speed_ec = np.mean(np.sqrt(COM_speeds_x[start_5_ec:end_ec]**2 + COM_speeds_z[start_5_ec:end_ec]**2))
                c7_x_ec = marker_dict['markers']['C7_study'][:,0][start_5_ec:end_ec]
                c7_z_ec = marker_dict['markers']['C7_study'][:,2][start_5_ec:end_ec]
                c7_std_ec = (np.sqrt(np.std(c7_x_ec)**2 + np.std(c7_z_ec)**2)) / marker_dict['markers']['C7_study'][:,1][start_ec]
            else:
                mean_com_speed_ec = np.nan
                c7_std_ec = np.nan
            standing_ec_features = {"time": time, "time_eo": time_eo, "time_ec": time_ec, "mean_com_speed_eo": mean_com_speed_eo, "mean_com_speed_ec": mean_com_speed_ec, "c7_std_eo":c7_std_eo, "c7_std_ec": c7_std_ec }
            filepath = 'features/standing_ec_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(standing_ec_features, json_file)
    print('Finished')