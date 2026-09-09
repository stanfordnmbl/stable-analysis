## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def find_segments(toe_r_x_vel, toe_l_x_vel, thresholds=np.arange(0.7, 0.29, -0.1)):
    n = len(toe_r_x_vel)
    for vel_thresh in thresholds:
        try:
            i = 0
            while toe_r_x_vel[i] < vel_thresh: i +=1
            front_r_start = i
            while toe_r_x_vel[i] > -vel_thresh: i +=1
            while toe_l_x_vel[i] < vel_thresh: i+=1
            front_r_end = i
            front_l_start = i 
            while toe_l_x_vel[i] > -vel_thresh: i+=1
            while toe_r_x_vel[i] > -vel_thresh: i+=1
            front_l_end = i
            back_r_start = i
            while toe_r_x_vel[i] < vel_thresh: i+=1
            while toe_l_x_vel[i] > - vel_thresh: i+=1
            back_r_end = i
            back_l_start = i
            while toe_l_x_vel[i] < vel_thresh: i+=1
            while toe_r_x_vel[i] > -vel_thresh: i+=1
            back_l_end = i
            cross_r_start = i
            while toe_r_x_vel[i] < vel_thresh: i+=1
            while toe_l_x_vel[i] >-vel_thresh: i+=1
            cross_r_end = i
            cross_l_start = i
            while toe_l_x_vel[i] < vel_thresh: i+=1
            while i < len(toe_r_x_vel): i+=1
            cross_l_end = i
            return {
                        "vel_thresh": vel_thresh,
                        "front_r_start": front_r_start, "front_r_end": front_r_end,
                        "front_l_start": front_l_start, "front_l_end": front_l_end,
                        "back_r_start": back_r_start,   "back_r_end": back_r_end,
                        "back_l_start": back_l_start,   "back_l_end": back_l_end,
                        "cross_r_start": cross_r_start, "cross_r_end": cross_r_end,
                        "cross_l_start": cross_l_start, "cross_l_end": cross_l_end,
                           }      
        except IndexError:
            continue
    raise RuntimeError(f"Failed to find windows for thresholds {list(thresholds)}")
    
def get_features_y_balance():
    print('Running y_balance...')
    record_ids = ['42', '68', '140', '144', '167']
    for i in range(5):
            record_id = record_ids[i]
            y_balance_features = {}  
            trial_names = ["y-balance"]
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
        
            # EXTRACT MARKERS
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # GET TEMPORAL CUES FOR RECORDING AND WRIST PEAK #############################
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.5) # changed from 0.75
            wrist_peak_idx = wrist_peak_idx[0]
        
            # LOW PASS FILTER #########################################################
            fs = 60
            cutoff = 2
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            ## SEGMENTATION ###############################################################    
            calc_r_x = marker_dict['markers']['r_calc_study'][:,0]
            calc_r_y = marker_dict['markers']['r_calc_study'][:,1]
            calc_r_z = marker_dict['markers']['r_calc_study'][:,2]
            calc_l_x = marker_dict['markers']['L_calc_study'][:,0]
            calc_l_y = marker_dict['markers']['L_calc_study'][:,1]
            calc_l_z = marker_dict['markers']['L_calc_study'][:,2]
            
            toe_r_x = marker_dict['markers']['r_toe_study'][:,0]
            #toe_r_y = marker_dict['markers']['r_toe_study'][:,1]
            toe_r_z = marker_dict['markers']['r_toe_study'][:,2]
            toe_l_x = marker_dict['markers']['L_toe_study'][:,0]
            #toe_l_y = marker_dict['markers']['L_toe_study'][:,1]
            toe_l_z = marker_dict['markers']['L_toe_study'][:,2]
        
            psis_r_x = marker_dict['markers']['r.PSIS_study'][:,0]
            psis_r_y = marker_dict['markers']['r.PSIS_study'][:,1]
            psis_r_z = marker_dict['markers']['r.PSIS_study'][:,2]
            
            psis_l_x = marker_dict['markers']['L.PSIS_study'][:,0]
            psis_l_y = marker_dict['markers']['L.PSIS_study'][:,1]
            psis_l_z = marker_dict['markers']['L.PSIS_study'][:,2]
        
            toe_dist = np.sqrt((toe_r_x - toe_l_x)**2 + (toe_r_z - toe_l_z)**2)
            
            toe_r_x_vel_raw = np.zeros((len(toe_r_x)))
            for t in range(len(toe_r_x)-1): toe_r_x_vel_raw[t] = ((toe_r_x[t+1] - toe_r_x[t])*60)
            toe_l_x_vel_raw = np.zeros((len(toe_l_x)))
            for t in range(len(toe_l_x)-1): toe_l_x_vel_raw[t] = ((toe_l_x[t+1] - toe_l_x[t])*60)
            
            toe_r_x_vel = filtfilt(b,a, toe_r_x_vel_raw)
            toe_l_x_vel = filtfilt(b,a, toe_l_x_vel_raw)
    
            # SEGMENTATION ###########################################
            segments = find_segments(toe_r_x_vel, toe_l_x_vel)
            
            front_r_start  = segments["front_r_start"];  front_r_end  = segments["front_r_end"]
            front_l_start  = segments["front_l_start"];  front_l_end  = segments["front_l_end"]
            back_r_start   = segments["back_r_start"];   back_r_end   = segments["back_r_end"]
            back_l_start   = segments["back_l_start"];   back_l_end   = segments["back_l_end"]
            cross_r_start  = segments["cross_r_start"];  cross_r_end  = segments["cross_r_end"]
            cross_l_start  = segments["cross_l_start"];  cross_l_end  = segments["cross_l_end"]

            # PEAKS #####################################################
            front_r_values = toe_dist[front_r_start:front_r_end][toe_r_x[front_r_start:front_r_end] > toe_r_x[wrist_peak_idx]+0.05]
            front_r_max= np.max(front_r_values)
            i = front_r_start
            while toe_dist[i] < front_r_max: i+=1
            
            front_l_values = toe_dist[front_l_start:front_l_end][toe_l_x[front_l_start:front_l_end] > toe_l_x[wrist_peak_idx]+0.05]
            front_l_max= np.max(front_l_values)
            i = front_l_start
            while toe_dist[i] < front_l_max: i+=1
            
            back_r_values = toe_dist[back_r_start:back_r_end][(toe_r_x[back_r_start:back_r_end] < toe_r_x[wrist_peak_idx]-0.05) & (toe_r_z[back_r_start:back_r_end] > toe_r_z[wrist_peak_idx])]
            back_r_max = np.max(back_r_values)
            i = back_r_start
            while toe_dist[i] < back_r_max: i+=1
            
            back_l_values = toe_dist[back_l_start:back_l_end][(toe_l_x[back_l_start:back_l_end] < toe_l_x[wrist_peak_idx]-0.05) & (toe_l_z[back_l_start:back_l_end] < toe_l_z[wrist_peak_idx])]
            back_l_max = np.max(back_l_values)
            i = back_l_start
            while toe_dist[i] < back_l_max: i+=1
            
            cross_r_values = toe_dist[cross_r_start:cross_r_end][(toe_r_x[cross_r_start:cross_r_end] < toe_r_x[wrist_peak_idx]-0.05) & (toe_r_z[cross_r_start:cross_r_end] < toe_r_z[wrist_peak_idx])]
            cross_r_max = np.max(cross_r_values)
            i = cross_r_start
            while toe_dist[i] < cross_r_max: i+=1
            
            cross_l_values = toe_dist[cross_l_start:cross_l_end][(toe_l_x[cross_l_start:cross_l_end] < toe_l_x[wrist_peak_idx]-0.05) & (toe_l_z[cross_l_start:cross_l_end] > toe_l_z[wrist_peak_idx])]
            cross_l_max = np.max(cross_l_values)
            i = cross_l_start
            while toe_dist[i] < cross_l_max: i+=1
            
            ## GET FEATURES ############################################################### 
            psis_calc_r = np.sqrt((psis_r_x - calc_r_x)**2 + (psis_r_y - calc_r_y)**2 + (psis_r_z - calc_r_z)**2)
            psis_calc_l = np.sqrt((psis_l_x - calc_l_x)**2 + (psis_l_y - calc_l_y)**2 + (psis_l_z - calc_l_z)**2)
            
            front_r = front_r_max/np.max(psis_calc_r)
            front_l = front_l_max/np.max(psis_calc_l)
            back_r = back_r_max/np.max(psis_calc_r)
            back_l = back_l_max/np.max(psis_calc_l)
            cross_r = cross_r_max/np.max(psis_calc_r)
            cross_l = cross_l_max/np.max(psis_calc_l) 
            
            y_balance_features = {"front_reach_r": front_r, "front_reach_l": front_l, "back_reach_r": back_r, "back_reach_l": back_l, "cross_reach_r": cross_r, "cross_reach_l": cross_l}
            filepath = 'features/y_balance_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(y_balance_features, json_file)
    print('Finished')