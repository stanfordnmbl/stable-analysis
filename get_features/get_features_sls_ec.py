## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
import json


def get_features_sls_ec():
    print('Running sls_ec...')
    record_ids = ['42', '68', '140', '144', '167', '179']
    for i in range(6):
            record_id = record_ids[i]
            sls_ec_features = {}  
            trial_names = ["sls-ec"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
        
            # Get COM KINEMATICS
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0]
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
            # Get center of mass values, speeds, and accelerations.
            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
            COM_speeds_z = center_of_mass['speeds'][trial_name]['z']
            
            # GET MARKER DATA
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # FIND WRIST PEAK
            from scipy.signal import find_peaks, peak_widths
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idx[0]
            
            # CUES #####################################################################
            # 0:00 lower, hands on hips, 0:05 "go", 0:15 "eyes closed", 0:25 "neutral", 0:30 "other side go", 0:40 "eyes-closed", 0:50 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [0, 3, 14, 19, 22, 33, 38]
            cue_idx = [(time * sample_freq) for time in cue_times] + wrist_peak_idx
        
            # SEGMENTATION - RIGHT SIDE ############################################################
            time = marker_dict['time']
            l_toe_y = marker_dict['markers']['L_toe_study'][:,1]
            
            i = cue_idx[1]
            l_up_eo = i
            l_down_eo = i
            toe_l_ground = l_toe_y[i]
            while i <= cue_idx[2] and (l_down_eo - l_up_eo)<(0.5*sample_freq):
                while l_toe_y[i] < toe_l_ground + 0.01 and i<=cue_idx[2]: i += 1
                l_up_eo = i
                while l_toe_y[i] >= toe_l_ground + 0.01 and i<=cue_idx[2]: i += 1
                while l_toe_y[i+1] < l_toe_y[i] and i<=cue_idx[2]: i += 1
                l_down_eo = i
            
            i = cue_idx[2]
            while l_toe_y[i] < toe_l_ground + 0.01 and i<=cue_idx[3]: i += 1
            l_up_ec = i
            l_down_ec = i
            while i <= cue_idx[3] and (l_down_ec - l_up_ec)<(0.5*sample_freq):
                while l_toe_y[i] >= toe_l_ground + 0.01 and i<=cue_idx[3]: i += 1
                while l_toe_y[i+1] < l_toe_y[i] and i<=cue_idx[3]: i += 1
                l_down_ec = i
                i+=1
            
            
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            i = l_up_eo
            calc_r_start_eo = calc_r_z[i]
            while np.abs(calc_r_z[i] - calc_r_start_eo) < 0.05 and i <= cue_idx[2] : i +=1
            calc_r_moved_eo = i
            
            i = l_up_ec
            calc_r_start_ec = calc_r_z[i]
            while np.abs(calc_r_z[i] - calc_r_start_ec) < 0.05 and i <= cue_idx[3] : i +=1
            calc_r_moved_ec = i
    
            # SEGMENTATION - LEFT SIDE ############################################################
            r_toe_y = marker_dict['markers']['r_toe_study'][:,1]

            i = cue_idx[4]
            r_up_eo = i
            r_down_eo = i
            toe_r_ground = r_toe_y[i]
            # get first point that crosses position threshold and is within the velocity threshold
            while i <= cue_idx[5] and (r_down_eo - r_up_eo)<(0.5*sample_freq):
                while r_toe_y[i] < toe_r_ground + 0.01 and i<=cue_idx[5]: i += 1
                r_up_eo = i
                while r_toe_y[i] >= toe_r_ground + 0.01 and i<=cue_idx[5]: i += 1
                while r_toe_y[i+1] < r_toe_y[i] and i<=cue_idx[5]: i += 1
                r_down_eo = i
            
            i = cue_idx[5]
            while r_toe_y[i] < toe_r_ground + 0.01 and i<=cue_idx[6]: i += 1
            r_up_ec = i
            r_down_ec = i
            while i <= cue_idx[6] and (r_down_ec - r_up_ec)<(0.5*sample_freq):
                while r_toe_y[i] >= toe_r_ground + 0.01 and i<=cue_idx[6]: i += 1
                while r_toe_y[i+1] < r_toe_y[i] and i<=cue_idx[6]: i += 1
                r_down_ec = i
                i+=1
            
            
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]
            i = r_up_eo
            calc_l_start_eo = calc_l_z[i]
            while np.abs(calc_l_z[i] - calc_l_start_eo) < 0.05 and i <= cue_idx[5] : i +=1
            calc_l_moved_eo = i
            
            i = r_up_ec
            calc_l_start_ec = calc_l_z[i]
            while np.abs(calc_l_z[i] - calc_l_start_ec) < 0.05 and i <= cue_idx[6] : i +=1
            calc_l_moved_ec = i
    
            # start and end indices 
            r_start_eo = l_up_eo
            r_end_eo = min(l_down_eo, calc_r_moved_eo)
            r_start_ec = l_up_ec
            r_end_ec = min(l_down_ec, calc_r_moved_ec)
            l_start_eo = r_up_eo
            l_end_eo = min(r_down_eo, calc_l_moved_eo)
            l_start_ec = r_up_ec
            l_end_ec = min(r_down_ec, calc_l_moved_ec)
    
            
            # FEATURES
            time_r_eo = min(time[r_end_eo] - time[r_start_eo], 10)
            time_r_ec = time[r_end_ec] - time[r_start_ec]
            
            time_l_eo = min(time[l_end_eo] - time[l_start_eo], 10)
            time_l_ec = time[l_end_ec] - time[l_start_ec]
            
            mean_com_speed_r_eo = np.mean(np.sqrt(COM_speeds_x[r_start_eo:r_end_eo]**2 + COM_speeds_z[r_start_eo:r_end_eo]**2))
            mean_com_speed_r_ec = np.mean(np.sqrt(COM_speeds_x[r_start_ec:r_end_ec]**2 + COM_speeds_z[r_start_ec:r_end_ec]**2))
            
            mean_com_speed_l_eo = np.mean(np.sqrt(COM_speeds_x[l_start_eo:l_end_eo]**2 + COM_speeds_z[l_start_eo:l_end_eo]**2))
            mean_com_speed_l_ec = np.mean(np.sqrt(COM_speeds_x[l_start_ec:l_end_ec]**2 + COM_speeds_z[l_start_ec:l_end_ec]**2))
            
            c7_x_r_eo = marker_dict['markers']['C7_study'][:,0][r_start_eo:r_end_eo]
            c7_z_r_eo = marker_dict['markers']['C7_study'][:,2][r_start_eo:r_end_eo]
            c7_std_r_eo = (np.sqrt(np.std(c7_x_r_eo)**2 + np.std(c7_z_r_eo)**2)) / marker_dict['markers']['C7_study'][:,1][r_start_eo]
            
            c7_x_r_ec = marker_dict['markers']['C7_study'][:,0][r_start_ec:r_end_ec]
            c7_z_r_ec = marker_dict['markers']['C7_study'][:,2][r_start_ec:r_end_ec]
            c7_std_r_ec = (np.sqrt(np.std(c7_x_r_ec)**2 + np.std(c7_z_r_ec)**2)) / marker_dict['markers']['C7_study'][:,1][r_start_ec]
            
            c7_x_l_eo = marker_dict['markers']['C7_study'][:,0][l_start_eo:l_end_eo]
            c7_z_l_eo = marker_dict['markers']['C7_study'][:,2][l_start_eo:l_end_eo]
            c7_std_l_eo = (np.sqrt(np.std(c7_x_l_eo)**2 + np.std(c7_z_l_eo)**2)) / marker_dict['markers']['C7_study'][:,1][l_start_eo]
            
            c7_x_l_ec = marker_dict['markers']['C7_study'][:,0][l_start_ec:l_end_ec]
            c7_z_l_ec = marker_dict['markers']['C7_study'][:,2][l_start_ec:l_end_ec]           
            c7_std_l_ec = (np.sqrt(np.std(c7_x_l_ec)**2 + np.std(c7_z_l_ec)**2)) / marker_dict['markers']['C7_study'][:,1][l_start_ec]
    
    
            sls_ec_features = {"time_r_eo": time_r_eo, "time_r_ec": time_r_ec, "time_l_eo": time_l_eo, "time_l_ec": time_l_ec, "mean_com_speed_r_eo": mean_com_speed_r_eo, "mean_com_speed_r_ec": mean_com_speed_r_ec, "mean_com_speed_l_eo": mean_com_speed_l_eo, "mean_com_speed_l_ec": mean_com_speed_l_ec, "c7_std_r_eo": c7_std_r_eo, "c7_std_r_ec": c7_std_r_ec, "c7_std_l_eo": c7_std_l_eo, "c7_std_l_ec": c7_std_l_ec}
            filepath = 'features/sls_ec_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(sls_ec_features, json_file)
    print('Finished')