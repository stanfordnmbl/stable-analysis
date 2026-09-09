# modified from https://github.com/opencap-org/opencap-processing

import os
import opensim
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy import signal

from utilsTRC import trc_2_dict
import numpy as np

REVERSE_MARKER_NAME_MAPPING = {}


class kinematics:
    
    def __init__(self, sessionDir, trialName, 
                 modelName=None,
                 lowpass_cutoff_frequency_for_coordinate_values=-1):
        
        self.lowpass_cutoff_frequency_for_coordinate_values = (
            lowpass_cutoff_frequency_for_coordinate_values)
        
        # Model.
        opensim.Logger.setLevelString('error')
        
        modelBasePath = os.path.join(sessionDir, 'OpenSimData', 'Model')
        
        # Check if this is a mono session (models stored in trial subfolders)
        # Check specifically for a subfolder matching the trial name
        isMono = False
        if os.path.exists(modelBasePath):
            trialModelPath = os.path.join(modelBasePath, trialName)
            if os.path.isdir(trialModelPath):
                isMono = True
        
        # Load model if specified, otherwise load the one that was on server
        if modelName is None:
            if isMono:
                # For mono sessions, look in the trial subfolder
                trialModelPath = os.path.join(modelBasePath, trialName)
                if os.path.exists(trialModelPath):
                    # Find .osim file in the trial subfolder
                    osimFiles = [f for f in os.listdir(trialModelPath) if f.endswith('.osim')]
                    if osimFiles:
                        modelPath = os.path.join(trialModelPath, osimFiles[0])
                    else:
                        raise Exception('No .osim file found in ' + trialModelPath)
                else:
                    raise Exception('Trial model folder does not exist: ' + trialModelPath)
            else:
                modelName = utils.get_model_name_from_metadata(sessionDir)
                modelPath = os.path.join(modelBasePath, modelName)
        else:
            if isMono:
                # For mono sessions, look in the trial subfolder
                trialModelPath = os.path.join(modelBasePath, trialName)
                if not modelName.endswith('.osim'):
                    modelName = modelName + '.osim'
                modelPath = os.path.join(trialModelPath, modelName)
            else:
                if not modelName.endswith('.osim'):
                    modelPath = os.path.join(modelBasePath, '{}.osim'.format(modelName))
                else:
                    modelPath = os.path.join(modelBasePath, modelName)
            
        # make sure model exists
        if not os.path.exists(modelPath):
            raise Exception('Model path: ' + modelPath + ' does not exist.')

        self.modelPath = modelPath
        self.model = opensim.Model(modelPath)
        self.model.initSystem()
        
        # Motion file with coordinate values.
        motionPath = os.path.join(sessionDir, 'OpenSimData', 'Kinematics',
                                  '{}.mot'.format(trialName))
        
        # Create time-series table with coordinate values.             
        self.table = opensim.TimeSeriesTable(motionPath)        
        tableProcessor = opensim.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())
        
        # Initialize the state trajectory. We will set it in other functions
        # if it is needed.
        self._stateTrajectory = None
        
        # Filter coordinate values.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            tableProcessor.append(
                opensim.TabOpLowPassFilter(
                    lowpass_cutoff_frequency_for_coordinate_values))

        # Convert in radians.
        self.table = tableProcessor.processAndConvertToRadians(self.model)
        
        # Trim if filtered.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            time_temp = self.table.getIndependentColumn()
            t_start = max(self.time[0], time_temp[0])
            t_end = min(self.time[-1], time_temp[-1])
            self.table.trim(
                time_temp[self.table.getNearestRowIndexForTime(t_start)],
                time_temp[self.table.getNearestRowIndexForTime(t_end)])
                
        # Compute coordinate speeds and accelerations and add speeds to table.        
        self.Qs = self.table.getMatrix().to_numpy()
        self.Qds = np.zeros(self.Qs.shape)
        self.Qdds = np.zeros(self.Qs.shape)
        columnAbsoluteLabels = list(self.table.getColumnLabels())
        for i, columnLabel in enumerate(columnAbsoluteLabels):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, self.Qs[:,i], k=3)
            # Coordinate speeds
            splineD1 = spline.derivative(n=1)
            self.Qds[:,i] = splineD1(self.time)
            # Coordinate accelerations.
            splineD2 = spline.derivative(n=2)
            self.Qdds[:,i] = splineD2(self.time)            
            # Add coordinate speeds to table.
            columnLabel_speed = columnLabel[:-5] + 'speed'
            self.table.appendColumn(
                columnLabel_speed, 
                opensim.Vector(self.Qds[:,i].flatten().tolist()))
            
        # Append missing muscle states to table.
        # Needed for StatesTrajectory.
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]
        existingLabels = self.table.getColumnLabels()
        for stateVariableNameStr in stateVariableNamesStr:
            if not stateVariableNameStr in existingLabels:
                vec_0 = opensim.Vector([0] * self.table.getNumRows())            
                self.table.appendColumn(stateVariableNameStr, vec_0)
                       
        # Number of muscles.
        self.nMuscles = 0
        self.forceSet = self.model.getForceSet()
        for i in range(self.forceSet.getSize()):        
            c_force_elt = self.forceSet.get(i)  
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                
        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName() 
                            for i in range(self.nCoordinates)]
            
        # Find rotational and translational coordinates.
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 2]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 1]
        
        # TODO: hard coded
        self.rootCoordinates = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        
        self.lumbarCoordinates = ['lumbar_extension', 'lumbar_bending', 
                                  'lumbar_rotation']
        
        self.armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                               'elbow_flex_r', 'pro_sup_r', 
                               'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                               'elbow_flex_l', 'pro_sup_l']
    
    # Only set the state trajectory when needed because it is slow.
    def stateTrajectory(self):
        if self._stateTrajectory is None:
            self._stateTrajectory = (
                opensim.StatesTrajectory.createFromStatesTable(
                    self.model, self.table))
        return self._stateTrajectory
    
    def get_marker_dict(self, session_dir, trial_name, 
                        lowpass_cutoff_frequency=-1):
        
        trcFilePath = os.path.join(session_dir,
                                   'MarkerData',
                                   '{}.trc'.format(trial_name))
        
        markerDict = trc_2_dict(trcFilePath)
        
        # Convert marker names from actual format to expected format (with _study suffix)
        if REVERSE_MARKER_NAME_MAPPING:
            converted_markers = {}
            # First pass: add markers that are already in correct format (prioritize these)
            for marker_name, marker_data in markerDict['markers'].items():
                if marker_name not in REVERSE_MARKER_NAME_MAPPING:
                    # Already in correct format or unknown marker - keep as-is
                    converted_markers[marker_name] = marker_data
            # Second pass: convert markers that need renaming (only if target doesn't exist)
            for marker_name, marker_data in markerDict['markers'].items():
                if marker_name in REVERSE_MARKER_NAME_MAPPING:
                    new_name = REVERSE_MARKER_NAME_MAPPING[marker_name]
                    # Only convert if the target name doesn't already exist
                    # (avoids overwriting markers already in correct format)
                    if new_name not in converted_markers:
                        converted_markers[new_name] = marker_data
            markerDict['markers'] = converted_markers
        
        if lowpass_cutoff_frequency > 0:
            markerDict['markers'] = {
                marker_name: lowPassFilter(self.time, data, lowpass_cutoff_frequency) 
                for marker_name, data in markerDict['markers'].items()}
        
        return markerDict

    def get_coordinate_values(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qs = np.zeros((self.Qs.shape))
            Qs[:, self.idxColumnTrLabels] = self.Qs[:, self.idxColumnTrLabels]
            Qs[:, self.idxColumnRotLabels] = (
                self.Qs[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qs = self.Qs
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: You are filtering the coordinate values a second time; coordinate values were filtered when creating your class object.")
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qs), axis=1)
        columns = ['time'] + self.columnLabels            
        self.coordinate_values = pd.DataFrame(data=data, columns=columns)
        
        return self.coordinate_values
    
    def get_coordinate_speeds(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qds = np.zeros((self.Qds.shape))
            Qds[:, self.idxColumnTrLabels] = (
                self.Qds[:, self.idxColumnTrLabels])
            Qds[:, self.idxColumnRotLabels] = (
                self.Qds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qds = self.Qds
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qds = lowPassFilter(self.time, Qds, lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qds), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_speeds = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_speeds
    
    def compute_center_of_mass(self):        
        
        # Compute center of mass position and velocity.
        self.com_values = np.zeros((self.table.getNumRows(),3))
        self.com_speeds = np.zeros((self.table.getNumRows(),3))        
        for i in range(self.table.getNumRows()):            
            self.model.realizeVelocity(self.stateTrajectory()[i])
            self.com_values[i,:] = self.model.calcMassCenterPosition(
                self.stateTrajectory()[i]).to_numpy()
            self.com_speeds[i,:] = self.model.calcMassCenterVelocity(
                self.stateTrajectory()[i]).to_numpy()
            
    def get_center_of_mass_values(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_v = self.com_values
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_v = lowPassFilter(self.time, com_v, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_v), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_values = pd.DataFrame(data=data, columns=columns)
        
        return com_values
    
    def get_center_of_mass_speeds(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_s = self.com_speeds
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_s = lowPassFilter(self.time, com_s, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_s), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_speeds = pd.DataFrame(data=data, columns=columns)
        
        return com_speeds

def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    
    fs = 1/np.round(np.mean(np.diff(time)),16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)
    return dataFilt