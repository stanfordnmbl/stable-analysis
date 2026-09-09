## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_semi_tandem(): 
    print('Running semi_tandem...')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            semi_tandem_features = {}  
            trial_names = ["semi-tandem"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
    
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0] 
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
            
            # GET CENTER OF MASS VALUES, SPEEDS, ACCELERATIONS
            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
            COM_speeds_z = center_of_mass['speeds'][trial_name]['z']
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
            marker_time = marker_dict['time']
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idx[0]
        
            # CUES #####################################################################
            # 0:00 "lower", 0:01 "hands on hips", 0:03 "go right", 0:14 "neutral", 0:15 "other side go", 0:26 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [1, 2, 14, 16, 26.5]  
            cue_idx = [round(time * sample_freq) for time in cue_times] + wrist_peak_idx

            # LOW PASS FILTER #########################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################
            calc_r_x = marker_dict['markers']['r_calc_study'][:,0]
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            calc_l_x = marker_dict['markers']['L_calc_study'][:,0]
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]
            calc_dist = np.sqrt((calc_r_x - calc_l_x)**2 + (calc_r_z - calc_l_z)**2)
            
            toe_r_x = marker_dict['markers']['r_toe_study'][:,0]
            toe_r_z = marker_dict['markers']['r_toe_study'][:,2]
            toe_l_x = marker_dict['markers']['L_toe_study'][:,0]
            toe_l_z = marker_dict['markers']['L_toe_study'][:,2]
            toe_dist = np.sqrt((toe_r_x - toe_l_x)**2 + (toe_r_z - toe_l_z)**2)
            
            meta_r_x = marker_dict['markers']['r_5meta_study'][:,0]
            meta_r_z = marker_dict['markers']['r_5meta_study'][:,2]
            meta_l_x = marker_dict['markers']['L_5meta_study'][:,0]
            meta_l_z = marker_dict['markers']['L_5meta_study'][:,2]
            meta_dist = np.sqrt((meta_r_x - meta_l_x)**2 + (meta_r_z - meta_l_z)**2)
            
            least_squares = np.sqrt(calc_dist**2 + toe_dist**2 + meta_dist**2) #+mankle_dist**2
            dt = np.diff(marker_dict['time'])
            least_squares_der = np.hstack([0, np.diff(least_squares) / dt])
            least_squares_der_filt = filtfilt(b,a, least_squares_der)
        
            i = cue_idx[1]
            start_r = i
            end_r = i
            next_start = True
            # get first point that crosses position threshold and is within the velocity threshold
            while least_squares_der_filt[i] <= 0.1 and least_squares_der_filt[i] >= - 0.1 and i<=cue_idx[2]: i += 1
            while i <= cue_idx[2] and ((end_r - start_r)<(1*sample_freq) or (next_start == True and end_r - start_r<(5*sample_freq))):
                while (least_squares[i] > (least_squares[cue_idx[1]]) or least_squares_der_filt[i] > 0.1 or least_squares_der_filt[i] < - 0.1) and i <= cue_idx[2]: i += 1
                start_r = i
                while least_squares_der_filt[i] <= 0.1 and least_squares_der_filt[i] >= - 0.1 and i<=(start_r+600): i += 1
                end_r = i
                while least_squares_der_filt[i] > 0.1 or least_squares_der_filt[i] < - 0.1: i+=1
                next_start = least_squares[i] < least_squares[end_r]
    
            
            # if position is closer after next motion pause, use that one  
            i = cue_idx[3]
            start_l = i
            end_l = i
            next_start = True
            while least_squares_der_filt[i] <= 0.1 and least_squares_der_filt[i] >= - 0.1 and i<=cue_idx[4]: i += 1
            while i <= cue_idx[4] and ((end_l - start_l)<(1*sample_freq) or (next_start == True and end_l - start_l<(5*sample_freq))):
                while (least_squares[i] > (least_squares[cue_idx[3]]) or least_squares_der_filt[i] > 0.1 or least_squares_der_filt[i] < - 0.1) and i <= cue_idx[4]: i += 1
                start_l = i
                while least_squares_der_filt[i] <= 0.1 and least_squares_der_filt[i] >= - 0.1 and i<=(start_l+600): i += 1
                end_l = i
                while least_squares_der_filt[i] > 0.1 or least_squares_der_filt[i] < - 0.1: i+=1
                next_start = least_squares[i] < least_squares[end_l]
        
            ## GET FEATURES ###############################################################
            time_r = min(marker_time[end_r]-marker_time[start_r], 10)
            time_l = min(marker_time[end_l]-marker_time[start_l], 10)
    
            if time_r != 0:
                mean_com_speed_r = np.mean(np.sqrt(COM_speeds_x[start_r:end_r]**2 + COM_speeds_z[start_r:end_r]**2))
                c7_x_r = marker_dict['markers']['C7_study'][:,0][start_r:end_r]
                c7_z_r = marker_dict['markers']['C7_study'][:,2][start_r:end_r]
                c7_std_r = (np.sqrt(np.std(c7_x_r)**2 + np.std(c7_z_r)**2)) / marker_dict['markers']['C7_study'][:,1][start_r]
            else:
                mean_com_speed_r = np.nan
                c7_std_r = np.nan
            if time_l != 0:
                mean_com_speed_l = np.mean(np.sqrt(COM_speeds_x[start_l:end_l]**2 + COM_speeds_z[start_l:end_l]**2))
                c7_x_l = marker_dict['markers']['C7_study'][:,0][start_l:end_l]
                c7_z_l = marker_dict['markers']['C7_study'][:,2][start_l:end_l]
                c7_std_l = (np.sqrt(np.std(c7_x_l)**2 + np.std(c7_z_l)**2)) / marker_dict['markers']['C7_study'][:,1][start_l]
            else:
                mean_com_speed_l = np.nan
                c7_std_l = np.nan
                   
            semi_tandem_features = {"time_r": time_r, "time_l": time_l, "mean_com_speed_r": mean_com_speed_r, "mean_com_speed_l": mean_com_speed_l, "c7_std_r": c7_std_r, "c7_std_l": c7_std_l,
                                    }
            filepath = 'features/semi_tandem_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(semi_tandem_features, json_file)
    print('Finished')