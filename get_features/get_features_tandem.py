## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_tandem():
    print('Running tandem...')
    record_ids = ['42', '68', '140', '144', '167', '179']
    for i in range(6):
            record_id = record_ids[i]
            tandem_features = {}  
            trial_names = ["tandem"]
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
        
            # EXTRACT COORDINATE VALUES, MARKERS
            coordinate_values = kinematics[trial_name].get_coordinate_values(in_degrees=True)
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
            marker_time = marker_dict['time']
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5)
            wrist_peak_idx = wrist_peak_idx[0]
        
            # CUES #####################################################################
            # 0:00 "lower", 0:02 "arms in dinner platter", 0:03 "turn right", 0:04 "step tandem", 0:15 "face front", 0:17 "turn left", 0.18 "step tandem", 0:29 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [1, 2, 4, 14, 14, 18, 28]
            cue_idx = [(time * sample_freq) for time in cue_times] + wrist_peak_idx
            
        # LOW PASS FILTER ################################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################    
            calc_r_x = marker_dict['markers']['r_calc_study'][:,0]
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            calc_l_x = marker_dict['markers']['L_calc_study'][:,0]
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]
            
            toe_r_x = marker_dict['markers']['r_toe_study'][:,0]
            toe_r_z = marker_dict['markers']['r_toe_study'][:,2]
            toe_l_x = marker_dict['markers']['L_toe_study'][:,0]
            toe_l_z = marker_dict['markers']['L_toe_study'][:,2]
            
            r_calc_l_toe = np.sqrt((calc_r_x - toe_l_x)**2 + (calc_r_z - toe_l_z)**2)
            l_calc_r_toe = np.sqrt((calc_l_x - toe_r_x)**2 + (calc_l_z - toe_r_z)**2)
            
            dt = np.diff(marker_dict['time'])
            r_calc_l_toe_der = np.hstack([0, np.diff(r_calc_l_toe) / dt])
            r_calc_l_toe_der_filt = filtfilt(b,a, r_calc_l_toe_der)
            l_calc_r_toe_der = np.hstack([0, np.diff(l_calc_r_toe) / dt])
            l_calc_r_toe_der_filt = filtfilt(b,a, l_calc_r_toe_der)
             
            i = cue_idx[1]
            start_r = i
            end_r = i
            next_start = True
            # get first point that crosses position threshold and is within the velocity threshold
            while coordinate_values['pelvis_rotation'][i] > -15: i+= 1
            while r_calc_l_toe_der_filt[i] <= 0.1 and r_calc_l_toe_der_filt[i] >= - 0.1 and i<=(start_r+600): i += 1
            while i <= (cue_idx[1]+(12*sample_freq)) and ((end_r - start_r)<60 or next_start == True):
                while (r_calc_l_toe[i] > (r_calc_l_toe[cue_idx[1]]-0.05) or r_calc_l_toe[i] > 0.25 or r_calc_l_toe_der_filt[i] > 0.1 or r_calc_l_toe_der_filt[i] < - 0.1) and i <= (start_r+600): i += 1
                start_r = i
                while r_calc_l_toe_der_filt[i] <= 0.1 and r_calc_l_toe_der_filt[i] >= - 0.1 and i<(start_r+600): i += 1
                end_r = i
                while r_calc_l_toe_der_filt[i] > 0.1 or r_calc_l_toe_der_filt[i] < - 0.1: i+=1
                next_start = r_calc_l_toe[i] < r_calc_l_toe[round((end_r + start_r)/2)]
    
            i = end_r + 150
            start_l = i
            end_l = i
            next_start = True
            while coordinate_values['pelvis_rotation'][i] < 20: i+= 1
            while l_calc_r_toe_der_filt[i] <= 0.1 and l_calc_r_toe_der_filt[i] >= - 0.1 and i<=(start_l+600): i += 1
            while i <= (cue_idx[1]+(26*sample_freq)) and ((end_l - start_l)<60 or next_start == True):
                while (l_calc_r_toe[i] > (l_calc_r_toe[cue_idx[1]]-0.05) or l_calc_r_toe[i] > 0.25 or  l_calc_r_toe_der_filt[i] > 0.1 or l_calc_r_toe_der_filt[i] < - 0.1) and i <= (start_l+600): i += 1
                start_l = i
                while l_calc_r_toe_der_filt[i] <= 0.1 and l_calc_r_toe_der_filt[i] >= - 0.1 and i<(start_l+600): i += 1
                end_l = i
                while l_calc_r_toe_der_filt[i] > 0.1 or l_calc_r_toe_der_filt[i] < - 0.1: i +=1
                next_start = l_calc_r_toe[i] < l_calc_r_toe[round((end_l + start_l)/2)]
        
            ## GET FEATURES ###############################################################    
            time_r = marker_time[end_r]-marker_time[start_r]
            time_l = marker_time[end_l]-marker_time[start_l]
    
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
            
            tandem_features = {"time_r": time_r, "time_l": time_l, "mean_com_speed_r": mean_com_speed_r, "mean_com_speed_l": mean_com_speed_l, "c7_std_r": c7_std_r, "c7_std_l": c7_std_l}
            filepath = 'features/tandem_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(tandem_features, json_file)
    print('Finished')