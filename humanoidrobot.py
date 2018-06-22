"""""""""""""""""""""""""""""""""""""""""""""
DESCRIPTION:
    This file contains the HumanoidRobot class for use in performing SLAM
        on a humanoid robot.  This class provides all the robot parameters 
        necessary to get the transform matricies for FastSLAM.
    The HumanoidRobot class contains a FastSLAM object along with functions
        that interface with the FastSLAM object.
    
"""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from fastslam import *
from utils.load_data import *
from utils.misc import *


class HumanoidRobot():
    
    def __init__(self, sensor_head_dist = 0.15, head_body_dist = 0.33, 
                 floor_body_dist = 0.93, N_eff_threshold = 5):
        # Store relevant parameters 
        self.head_body_dist = head_body_dist
        self.sensor_head_dist = sensor_head_dist
        self.floor_body_dist = floor_body_dist
        self.N_eff_threshold = N_eff_threshold
        
        # Create FastSLAM Object (can be overwritten later if necessary)
        self.fastslam = FastSLAM(x_range = (-20,20), y_range = (-20,20), map_res = 0.05,
                 dtheta_res = 1, pos_sweep_ind = np.arange(-4,5,1), 
                 lidar_angles = np.arange(-135,135.25,0.25), lidar_range = (0.1,10),
                 lidar_sweep_ind = np.arange(-4,5,1), 
                 num_particles = 50, x_sigma = 0.001, y_sigma = 0.001, theta_sigma = 0.015)
        
        # Get camera extrinsics and calibration matrices
        self.extRGBD = getExtrinsics_IR_RGB()
        self.IRCalib = getIRCalib()
        self.RGBCalib = getRGBCalib()
        
        # Get static transformation matrices
        self.T_h2l = get_T(0,0,0,0,0,sensor_head_dist)
        
        # Initialize running pose running pose
        self.T_w2b_last = get_T(0,0,0,0,0,0)
        
    def setup_map(self, l_scan, l_pose_x, l_pose_y, l_pose_yaw, h_pitch, h_yaw, ts):
        T_b2h = get_T(0,h_pitch,h_yaw,0,0,self.head_body_dist)
        T_b2l = np.matmul(T_b2h,self.T_h2l)
        init_pose = get_T(0,0,0,0,0,0)
        self.fastslam.update_map(l_scan, init_pose, T_b2l)
    
    def predict(self, l_scan, l_pose_x, l_pose_y, l_pose_yaw, h_pitch, h_yaw, ts):
        # get odometries
        T_b2h = get_T(0,h_pitch,h_yaw,0,0,self.head_body_dist)
        T_b2l = np.matmul(T_b2h,self.T_h2l)
        T_w2l = get_T(0,0,l_pose_yaw,l_pose_x,l_pose_y,0)
        T_w2b = np.matmul(T_w2l,np.linalg.inv(T_b2l))
        
        #predict
        T_w2b_best = self.fastslam.predict(self.T_w2b_last, T_w2b)
        self.T_w2b_last = T_w2b
        
        return T_w2b_best
        
    def predict_and_update(self, l_scan, l_pose_x, l_pose_y, l_pose_yaw, h_pitch, h_yaw, ts):
        # get odometries
        T_b2h = get_T(0,h_pitch,h_yaw,0,0,self.head_body_dist)
        T_b2l = np.matmul(T_b2h,self.T_h2l)
        T_w2l = get_T(0,0,l_pose_yaw,l_pose_x,l_pose_y,0)
        T_w2b = np.matmul(T_w2l,np.linalg.inv(T_b2l))
        
        #predict
        T_w2b_best = self.fastslam.predict(self.T_w2b_last, T_w2b)
        self.T_w2b_last = T_w2b
        
        #update particles
        T_w2b_best = self.fastslam.update(l_scan, T_b2l, self.floor_body_dist)
        
        #update map
        self.fastslam.update_map(l_scan, T_w2b_best, T_b2l)
        
        #check if resampling is needed
        if self.fastslam.get_N_eff() < self.N_eff_threshold:
            self.fastslam.resample()
            
        return T_w2b_best
        
    def get_map(self):
        return self.fastslam.map
        