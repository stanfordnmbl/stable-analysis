## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_sls_rtt():
    print('Running sls_rtt...')
    record_ids = ['42', '68', '140', '144']
    for i in range(4):
            record_id = record_ids[i]
            sls_rtt_features = {}  
            trial_names = ["sls-rtt"]
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
            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
               
            COM_speeds_x = center_of_mass['speeds'][trial_name]['x']
            COM_speeds_z = center_of_mass['speeds'][trial_name]['z']
        
            # EXTRACT COORDINATE VALUES, MARKERS
            coordinate_values = kinematics[trial_name].get_coordinate_values(in_degrees=True)
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idx[0]
        
            # CUES #####################################################################
            # 0:00 "lower", 0:02 "hands on hips", 0:03 "go", 0:05 "rise", 0:10 "neutral", 0:12 "otherside go", 0.14 "rise", 0:19 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [2, 3, 5, 10, 12, 14, 19]
            cue_idx = [(time * sample_freq) for time in cue_times] + wrist_peak_idx

            # LOW PASS FILTER #########################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################    
            calc_r_y = marker_dict['markers']['r_calc_study'][:,1]
            calc_l_y = marker_dict['markers']['L_calc_study'][:,1]
            toe_r_y = marker_dict['markers']['r_toe_study'][:,1]
            toe_l_y = marker_dict['markers']['L_toe_study'][:,1]
              
            time = marker_dict['time']

            calc_r_filt = filtfilt(b,a, calc_r_y)
            calc_l_filt = filtfilt(b,a,calc_l_y)
        
            # get foot off the ground indeces
            i = cue_idx[1]
            start_r_sls = i
            end_r_sls = i
            toe_l_ground = toe_l_y[i]
            # get first point that crosses position threshold and is within the velocity threshold
            while i <= cue_idx[3] and (end_r_sls - start_r_sls)<30:
                while toe_l_y[i] < toe_l_ground + 0.01 and i<=cue_idx[3]: i += 1
                start_r_sls = i
                while toe_l_y[i] >= toe_l_y[start_r_sls] and i<=start_r_sls+420: i += 1
                while toe_l_y[i+1] < toe_l_y[i] and i<=start_r_sls+420: i += 1
                end_r_sls = i
            
            i = cue_idx[4]
            start_l_sls = i
            end_l_sls = i
            toe_r_ground = toe_r_y[i]
            # get first point that crosses position threshold and is within the velocity threshold
            while i <= cue_idx[6] and (end_l_sls - start_l_sls)<30:
                while toe_r_y[i] < toe_r_ground + 0.01 and i<=cue_idx[6]: i += 1
                start_l_sls = i
                while toe_r_y[i] >= toe_r_y[start_l_sls] and i<=start_l_sls+420: i += 1
                while toe_r_y[i+1] < toe_r_y[i] and i<=start_l_sls+420: i += 1
                end_l_sls = i
        
            # get heel up indexes
            i = start_r_sls+30
            start_r_calc = i
            end_r_calc = i
            calc_r_ground = calc_r_filt[i]
            
            peak_calc_r = np.max(calc_r_filt[start_r_sls:end_r_sls]) - calc_r_ground
            peak_calc_r_thresh = (0.25 * peak_calc_r) + calc_r_ground
            # get first point that crosses position threshold and is within the velocity threshold
            while i <= cue_idx[3] and (end_r_calc - start_r_calc)<20:
                while calc_r_filt[i] < peak_calc_r_thresh and i<=cue_idx[3]: i += 1
                while calc_r_filt[i] > calc_r_filt[i-1] or calc_r_filt[i] > peak_calc_r_thresh: i-=1
                start_r_calc = i
                while calc_r_filt[i] < peak_calc_r_thresh: i+=1
                while calc_r_filt[i] < peak_calc_r + calc_r_ground and calc_r_filt[i] >= peak_calc_r_thresh and i<=cue_idx[3]: i += 1
                while calc_r_filt[i] >= peak_calc_r_thresh and i<=start_r_calc+300: i += 1
                while calc_r_filt[i] > calc_r_filt[i+1] and i<=start_r_calc+300: i += 1
                end_r_calc = i           
            
            i = start_l_sls+30
            start_l_calc = i
            end_l_calc = i
            calc_l_ground = calc_l_filt[i]
            
            peak_calc_l = np.max(calc_l_filt[start_l_sls:end_l_sls]) - calc_l_ground
            peak_calc_l_thresh = (0.25 * peak_calc_l) + calc_l_ground
            # get first point that crosses position threshold and is within the velocity threshold
            while i <= cue_idx[6] and (end_l_calc - start_l_calc)<20:
                while calc_l_filt[i] < peak_calc_l_thresh and i<=cue_idx[6]: i += 1
                while calc_l_filt[i] > calc_l_filt[i-1] or calc_l_filt[i] > peak_calc_l_thresh: i-=1
                start_l_calc = i
                while calc_l_filt[i] < peak_calc_l_thresh: i+=1
                while calc_l_filt[i] < peak_calc_l + calc_l_ground and calc_l_filt[i] >= peak_calc_l_thresh and i<=cue_idx[6]: i += 1
                while calc_l_filt[i] >= peak_calc_l_thresh and i<=start_l_calc+300: i += 1
                while calc_l_filt[i] > calc_l_filt[i+1] and i<=start_l_calc+300: i += 1
                end_l_calc = i
    
            start_r = min([start_r_calc, end_r_sls])
            end_r = min([end_r_calc, end_r_sls])
            start_l = min([start_l_calc, end_l_sls])
            end_l = min([end_l_calc, end_l_sls])
            ## GET FEATURES ###############################################################
            time_r = min(time[end_r] - time[start_r], 5)
            time_l = min(time[end_l] - time[start_l], 5)
                    
            mean_com_speed_r = np.mean(np.sqrt(COM_speeds_x[start_r:end_r]**2 + COM_speeds_z[start_r:end_r]**2))
            mean_com_speed_l = np.mean(np.sqrt(COM_speeds_x[start_l:end_l]**2 + COM_speeds_z[start_l:end_l]**2))
                    
            c7_x_r = marker_dict['markers']['C7_study'][:,0][start_r:end_r]
            c7_z_r = marker_dict['markers']['C7_study'][:,2][start_r:end_r]
            c7_std_r = (np.sqrt(np.std(c7_x_r)**2 + np.std(c7_z_r)**2)) / marker_dict['markers']['C7_study'][:,1][start_r]
            
            c7_x_l = marker_dict['markers']['C7_study'][:,0][start_l:end_l]
            c7_z_l = marker_dict['markers']['C7_study'][:,2][start_l:end_l]
            c7_std_l = (np.sqrt(np.std(c7_x_l)**2 + np.std(c7_z_l)**2)) / marker_dict['markers']['C7_study'][:,1][start_l]
    
            peak_pf_r = np.max(-coordinate_values['ankle_angle_r'][start_r:end_r])
            if np.isnan(peak_pf_r): peak_pf_r = 0
            peak_pf_l = np.max(-coordinate_values['ankle_angle_l'][start_l_calc:end_l_calc])
            if np.isnan(peak_pf_l): peak_pf_l = 0
    
            sls_rtt_features = {"time_r": time_r, "time_l": time_l, "mean_com_speed_r": mean_com_speed_r, "mean_com_speed_l": mean_com_speed_l, "c7_std_r": c7_std_r, "c7_std_l": c7_std_l, "peak_pf_r": peak_pf_r, "peak_pf_l": peak_pf_l,
                                    }
            filepath = 'features/sls_rtt_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(sls_rtt_features, json_file)
    print('Finished')