## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_rtt():
    print('Running rtt...')
    record_ids = ['42', '68', '140', '144', '167', '179', '205']
    for i in range(7):
            record_id = record_ids[i]
            rtt_features = {}  
            trial_names = ["rtt"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"

            trial_name = trial_names[0]
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
        
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
        
            # GET CENTER OF MASS VALUES, SPEEDS, ACCELERATIONS
            center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
        
            COM_time = center_of_mass['values'][trial_name]['time']
            COM_values_x = center_of_mass['values'][trial_name]['x']
            COM_values_y = center_of_mass['values'][trial_name]['y']    
        
            # EXTRACT COORDINATE VALUES, MARKERS
            coordinate_values = kinematics[trial_name].get_coordinate_values(in_degrees=True)
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idxs,_ = find_peaks(signal, prominence=0.75)
            wrist_peak_idx = wrist_peak_idxs[0]
            if record_id == "68": wrist_peak_idx = wrist_peak_idxs[1]
        
            # CUES #####################################################################
            # 0:00 lower, 0:02 hands on hips, 0:03 "go", 0:085 "neutral", 0:25 "neutral"
            sample_freq = round(1/marker_dict['time'][1])
            cue_times = [2, 3, 8]
            cue_idx = [(time * sample_freq) for time in cue_times] + wrist_peak_idx
        
            # LOW PASS FILTER #########################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################
            # SEGMENT 1 (Anticipatory)
            COM_values_x_filt = filtfilt(b,a, COM_values_x)
            dt = np.diff(COM_time)
            COM_x_filt_der = np.hstack([0, np.diff(COM_values_x_filt) / dt])
            COM_x_filt_der_der = np.hstack([0, np.diff(COM_x_filt_der) / dt])
            seg_1 = np.argmax(COM_x_filt_der_der[cue_idx[1]-(1*sample_freq):cue_idx[1]+(1*sample_freq)]) + cue_idx[1]-(1*sample_freq)
        
            # SEGMENT 2 (Rise), 3 (Hold), 4 (Lower)
            COM_values_y_filt = filtfilt(b, a, COM_values_y)
            COM_y_filt_der = np.hstack([0, np.diff(COM_values_y_filt) / dt])
            COM_y_filt_der_der = np.hstack([0, np.diff(COM_y_filt_der) / dt])
            seg_2 = np.argmax(COM_y_filt_der_der[seg_1+1:seg_1+(1*sample_freq)]) + seg_1+1
        
            peak_y_idx = np.argmax(COM_values_y[seg_1:-1])+seg_1
            peak_y = COM_values_y[peak_y_idx]
        
            # 5/50/95% RISE and 5/50% Lower
            COM_rise_total = (peak_y-(COM_values_y[seg_2]))
            COM_rise_05 = 0.05*COM_rise_total
            COM_rise_50 = 0.50*COM_rise_total
            COM_rise_95 = 0.95*COM_rise_total
        
            i = seg_2
            while COM_values_y[i] < COM_values_y[seg_2] + COM_rise_05 and i < cue_idx[2]: i = i+1
            idx_05 = i
            time_05 = COM_time[i]
        
            while COM_values_y[i] < COM_values_y[seg_2] + COM_rise_50 and i < cue_idx[2]: i = i+1
            idx_50 = i
            time_50 = COM_time[i]
        
            while COM_values_y[i] < COM_values_y[seg_2] + COM_rise_95 and i < cue_idx[2]: i = i+1
            idx_95 = i
            time_95 = COM_time[i]
        
            try:
              i = idx_50
              while COM_values_y[i] > COM_values_y[seg_2] + COM_rise_05: i = i+1
              idx_05_dwn = i
            except:
              i = idx_50
              COM_rise_total = (peak_y-(COM_values_y[seg_1]))
              COM_rise_05 = 0.05*COM_rise_total
              while COM_values_y[i] > COM_values_y[seg_1] + COM_rise_05: i = i+1
              idx_05_dwn = i
        
            while COM_values_y[i] < COM_values_y[seg_2] + COM_rise_50: i = i-1
            idx_50_dwn = i
        
            seg_3 = idx_50
            seg_4 = idx_50_dwn
            seg_4_end = idx_05_dwn
        
        
            ## GET FEATURES ###############################################################
            peak_pf_angle_r = np.max(-coordinate_values['ankle_angle_r'][seg_1:seg_4_end])
            peak_pf_angle_l = np.max(-coordinate_values['ankle_angle_l'][seg_1:seg_4_end])
            peak_pf = np.mean([peak_pf_angle_l, peak_pf_angle_r])
            peak_com_speed = np.max(COM_y_filt_der[seg_2:seg_3])   
            time = min(COM_time[seg_4] - COM_time[seg_3], 5)
            c7_x = marker_dict['markers']['C7_study'][:,0][seg_3:seg_4]
            c7_z = marker_dict['markers']['C7_study'][:,2][seg_3:seg_4]
            c7_std = (np.sqrt(np.std(c7_x)**2 + np.std(c7_z)**2)) / marker_dict['markers']['C7_study'][:,1][seg_3]
        
            rtt_features = {"peak_pf": peak_pf, "peak_com_speed": peak_com_speed, "time": time, "c7_std": c7_std}

            filepath = 'features/rtt_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(rtt_features, json_file)
    print('Finished')