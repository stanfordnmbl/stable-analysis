## IMPORT PACKAGES #############################################################
import os
import numpy as np
import utilsKinematics
from scipy.signal import find_peaks
import json
from scipy.signal import butter, filtfilt

def get_features_march():
    print('Running march...')
    record_ids = ['42', '68', '140', '144', '167', '179']
    for i in range(6):
            record_id = record_ids[i]
            march_features = {}  
            trial_names = ["march"]
            data_folder = os.path.join("../opencap_data", record_id)
            modelName = "LaiUhlrich2022_scaled"
        
            # Get center of mass kinematics.
            kinematics, center_of_mass = {}, {}
            center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
            trial_name = trial_names[0]
            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)
            # Get center of mass values, speeds, and accelerations.
            center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
            # GET COORDINATES AND MARKER DATA
            coordinate_values = kinematics[trial_name].get_coordinate_values(in_degrees=True)
            marker_dict = kinematics[trial_name].get_marker_dict(session_dir = data_folder , trial_name = trial_name)
        
            # FIND WRIST PEAK
            signal= marker_dict['markers']['r_lwrist_study'][:,1]
            wrist_peak_idx,_ = find_peaks(signal, prominence=0.50) # updated from 0.75
            wrist_peak_idx = wrist_peak_idx[0]
            sample_freq = round(1/marker_dict['time'][1])
            go_idx = wrist_peak_idx + round(2.5*sample_freq)
        
            # STEP SEGMENTATION #########################################################
            # Left Calcaneus Vertical Velocity
            dt = np.diff(marker_dict['time'])
            L_calc_vel = np.hstack([0, np.diff(marker_dict['markers']['L_toe_study'][:,1]) / dt])
            # Toe-off
            L_calc_diff_peaks_up,_ = find_peaks(L_calc_vel[go_idx:-1], prominence=max(L_calc_vel)*1/2)
            L_calc_diff_peaks_up = L_calc_diff_peaks_up + go_idx
        
            L_calc_TO = []
            for peak in L_calc_diff_peaks_up:
              i = 0
              while L_calc_vel[peak-i] > 0 and peak+i < len(L_calc_vel)-1:i += 1
              L_calc_TO.append(peak-i)
        
            # Heal-strike
            L_calc_diff_peaks_down,_ = find_peaks(-L_calc_vel[go_idx:-1], prominence=max(L_calc_vel)*1/2)
            L_calc_diff_peaks_down = L_calc_diff_peaks_down + go_idx
            
            L_calc_HS = []
            for peak in L_calc_diff_peaks_down:
              i = 0
              while L_calc_vel[peak+i] < 0: i+= 1
              L_calc_HS.append(peak+i)
            
              ## CLEAN UP SEGMENTATION ##########################################################
            neighbor = []
            for i in range(len(L_calc_TO)-1):
                neighbor.append(L_calc_TO[i+1] - L_calc_TO[i]) 
            threshold = 2/3 * np.mean(neighbor)
            indices_to_remove = []
            for i in range(len(neighbor)):
                if neighbor[i] < threshold:
                    if i == 0:  # First neighbor, no left neighbor
                        if neighbor[i+1] < threshold: indices_to_remove.append(i+1)
                        else: indices_to_remove.append(i)
                    elif i == len(neighbor) - 1:  # Last neighbor, no right neighbor
                        indices_to_remove.append(i+1)
                    else:
                        if neighbor[i-1] < neighbor[i+1]: indices_to_remove.append(i)
                        else: indices_to_remove.append(i+1)
            
            indices_to_remove = list(set(indices_to_remove))  # Remove duplicates and ensure unique indices
            L_calc_TO_filtered = [L_calc_TO[i] for i in range(len(L_calc_TO)) if i not in indices_to_remove]
            
            neighbor = []
            for i in range(len(L_calc_HS)-1):
                neighbor.append(L_calc_HS[i+1] - L_calc_HS[i])
            threshold = 2/3 * np.mean(neighbor)
            indices_to_remove = []
            for i in range(len(neighbor)):
                if neighbor[i] < threshold:
                    if i == 0:  # First neighbor, no left neighbor
                        if neighbor[i+1] < threshold: indices_to_remove.append(i+1)
                        else: indices_to_remove.append(i)
                    elif i == len(neighbor) - 1:  # Last neighbor, no right neighbor
                        indices_to_remove.append(i+1)
                    else:
                        if neighbor[i-1] < neighbor[i+1]: indices_to_remove.append(i)
                        else: indices_to_remove.append(i+1)
            
            indices_to_remove = list(set(indices_to_remove))  # Remove duplicates and ensure unique indices
            L_calc_HS_filtered = [L_calc_HS[i] for i in range(len(L_calc_HS)) if i not in indices_to_remove]
            
            # RIGHT####################################################################
            # vertical velocity
            dt = np.diff(marker_dict['time'])
            r_calc_vel = np.hstack([0, np.diff(marker_dict['markers']['r_toe_study'][:,1]) / dt])
        
            # Toe-off
            r_calc_diff_peaks_up,_ = find_peaks(r_calc_vel[go_idx:-1], prominence=max(r_calc_vel)*1/2)
            r_calc_diff_peaks_up += go_idx
        
            r_calc_TO = []
            for peak in r_calc_diff_peaks_up:
              i = 0
              while r_calc_vel[peak-i] > 0:i += 1
              r_calc_TO.append(peak-i)
        
            # Heal-strike
            r_calc_diff_peaks_down,_ = find_peaks(-r_calc_vel[go_idx:-1], prominence = max(r_calc_vel)*1/2)
            r_calc_diff_peaks_down += go_idx
              
            r_calc_HS = []
            for peak in r_calc_diff_peaks_down:
              i = 0
              while r_calc_vel[peak+i] < 0 and peak+i < len(r_calc_vel)-1: i+= 1
              r_calc_HS.append(peak+i)
            
            # CLEAN UP SEGMENTATION ########################################################
            neighbor = []
            for i in range(len(r_calc_TO)-1):
                neighbor.append(r_calc_TO[i+1] - r_calc_TO[i])
            threshold = 2/3 * np.mean(neighbor)
            indices_to_remove = []
            for i in range(len(neighbor)):
                if neighbor[i] < threshold:
                    if i == 0:  # First neighbor, no left neighbor
                        if neighbor[i+1] < threshold: indices_to_remove.append(i+1)
                        else: indices_to_remove.append(i)
                    elif i == len(neighbor) - 1:  # Last neighbor, no right neighbor
                        indices_to_remove.append(i+1)
                    else:
                        if neighbor[i-1] < neighbor[i+1]: indices_to_remove.append(i)
                        else: indices_to_remove.append(i+1)
            
            indices_to_remove = list(set(indices_to_remove))  # Remove duplicates and ensure unique indices
            r_calc_TO_filtered = [r_calc_TO[i] for i in range(len(r_calc_TO)) if i not in indices_to_remove]
            
            neighbor = []
            for i in range(len(r_calc_HS)-1):
                neighbor.append(r_calc_HS[i+1] - r_calc_HS[i])
            threshold = 2/3 * np.mean(neighbor)
            indices_to_remove = []
            for i in range(len(neighbor)):
                if neighbor[i] < threshold:
                    if i == 0:  # First neighbor, no left neighbor
                        if neighbor[i+1] < threshold: indices_to_remove.append(i+1)
                        else: indices_to_remove.append(i)
                    elif i == len(neighbor) - 1:  # Last neighbor, no right neighbor
                        indices_to_remove.append(i+1)
                    else:
                        if neighbor[i-1] < neighbor[i+1]: indices_to_remove.append(i)
                        else: indices_to_remove.append(i+1)
            
            indices_to_remove = list(set(indices_to_remove))  # Remove duplicates and ensure unique indices
            r_calc_HS_filtered = [r_calc_HS[i] for i in range(len(r_calc_HS)) if i not in indices_to_remove]
    
            # UPDATE SEGMENTATIONS AND TRIM
            L_calc_TO = L_calc_TO_filtered
            L_calc_HS = L_calc_HS_filtered
            r_calc_TO = r_calc_TO_filtered
            r_calc_HS = r_calc_HS_filtered
            
            if len(L_calc_TO) > 25: L_calc_TO = L_calc_TO[0:25]
            if len(L_calc_HS) > 25: L_calc_HS = L_calc_HS[0:25]
            if len(r_calc_TO) > 25: r_calc_TO = r_calc_TO[0:25]
            if len(r_calc_HS) > 25: r_calc_HS = r_calc_HS[0:25]
    
            # check if L_TO, L_HS, R_TO, R_HS or R_TO, R_HS, L_TO_L_HS is in the right order. Otherwise, adjust the TO to be 1 index past the HS
            # if HS before TO:
            if L_calc_HS[0] < L_calc_TO[0]: L_calc_HS.pop(0)
            if r_calc_HS[0] < r_calc_TO[0]: r_calc_HS.pop(0)
                
            # if lifting left leg first
            if L_calc_TO[0] < r_calc_TO[0]:
              for i in range(min(len(L_calc_TO)-1, len(r_calc_HS)-1)): ## Added a -1
                if r_calc_TO[i] <= L_calc_HS[i]: r_calc_TO[i] = L_calc_HS[i] +1
                if L_calc_TO[i+1] <= r_calc_HS[i]: L_calc_TO[i+1] = r_calc_HS[i] +1
        
            if r_calc_TO[0] < L_calc_TO[0]:
              for i in range(min(len(r_calc_TO) -1, len(L_calc_HS)-1)): ## Added a -1
                if L_calc_TO[i] <= r_calc_HS[i]: L_calc_TO[i] = r_calc_HS[i] +1
                if r_calc_TO[i+1] <= L_calc_HS[i]: r_calc_TO[i+1] = L_calc_HS[i] + 1
        
            # GET START AND FINISH INDEXES
            start_idx = min([L_calc_TO[0], r_calc_TO[0]])
            finish_idx = max([L_calc_HS[-1], r_calc_HS[-1]])
        
            # LOW PASS FILTER
            fs = 1/marker_dict['time'][1]
            cutoff = 0.25
            order = 4
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        
            # PELVIS ROTATION ###############################################
            pelvis_rot_filt = filtfilt(b,a, coordinate_values['pelvis_rotation'])
            rotation_change = pelvis_rot_filt[finish_idx] - pelvis_rot_filt[start_idx]
            rotation_change_abs = np.abs(rotation_change) 
        
            # COM TRANSLATION #####################################################
            COM_values_x = center_of_mass['values'][trial_name]['x']
            COM_values_z = center_of_mass['values'][trial_name]['z']
        
            com_x_filt = filtfilt(b,a, COM_values_x)
            com_z_filt = filtfilt(b,a, COM_values_z)
            com_dx = com_x_filt[finish_idx] - com_x_filt[start_idx]
            com_dx_abs = np.abs(com_dx)
            com_dz = com_z_filt[finish_idx] - com_z_filt[start_idx]
            com_dz_abs = np.abs(com_dz)
        
            march_features = {"rotation_change_abs": rotation_change_abs, "com_dx_abs": com_dx_abs, "com_dz_abs": com_dz_abs}
            filepath = 'features/march_features_' + record_id + '.json'
            with open(filepath, 'w') as json_file:
              json.dump(march_features, json_file)
    print('Finished')