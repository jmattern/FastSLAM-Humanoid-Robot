"""""""""""""""""""""""""""""""""""""""""""""
DESCRIPTION:
    This file contains the RobotDataLoader class for use in loading data for 
        the humanoid robot.  This data loader object is intended to mimic 
        generic time series data loader objects so that HumanoidRobot class can 
        be applied to different systems.
    
"""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from utils.load_data import *

class RobotDataLoader():
    
    def __init__(self, joint_path, lidar_path):
        # load in the data to memory
        self.joint_path = joint_path
        self.lidar_path = lidar_path
        self.joint_data = get_joint(joint_path)
        self.lidar_data = get_lidar(lidar_path)
        
        # load in required lidar data
        ind = np.arange(len(self.lidar_data))
        self.lidar_data_pose = np.array(list(map(lambda x : self.lidar_data[x]['pose'][0],ind)))
        self.lidar_data_scan = np.array(list(map(lambda x : self.lidar_data[x]['scan'][0],ind)))
        self.lidar_data_ts = np.array(list(map(lambda x : self.lidar_data[x]['t'][0],ind)))
        
        # load in required joint data
        self.head_data_angle = self.joint_data['head_angles']
        self.head_data_ts = self.joint_data['ts']
        
        # Initialize the sequential index to be 0
        self.current_idx = 0
    
    
    def __len__(self):
        return self.lidar_data_pose.shape[0]
    
    def get_item(self, idx):
        # get lidar measurement
        ts = self.lidar_data_ts[idx]
        l_scan = self.lidar_data_scan[idx]
        l_pose_x = self.lidar_data_pose[idx,0]
        l_pose_y = self.lidar_data_pose[idx,1]
        l_pose_yaw = self.lidar_data_pose[idx,2]
        
        # get closest head pose measurement to chosen lidar measurement
        h_idx = np.argmin(np.abs(self.head_data_ts-ts))
        h_pitch = self.head_data_angle[1,h_idx]
        h_yaw = self.head_data_angle[0,h_idx]
        
        return (l_scan, l_pose_x, l_pose_y, l_pose_yaw, h_pitch, h_yaw, ts)
    
    def get_next_item(self):
        self.current_idx += 1
        idx = self.current_idx
        # get lidar measurement
        ts = self.lidar_data_ts[idx]
        l_scan = self.lidar_data_scan[idx]
        l_pose_x = self.lidar_data_pose[idx,0]
        l_pose_y = self.lidar_data_pose[idx,1]
        l_pose_yaw = self.lidar_data_pose[idx,2]
        
        # get closest head pose measurement to chosen lidar measurement
        h_idx = np.argmin(np.abs(self.head_data_ts-ts))
        h_pitch = self.head_data_angle[1,h_idx]
        h_yaw = self.head_data_angle[0,h_idx]
        
        return (l_scan, l_pose_x, l_pose_y, l_pose_yaw, h_pitch, h_yaw, ts)
    
    def reset_idx(self):
        self.current_idx = 0
        
        
        
        