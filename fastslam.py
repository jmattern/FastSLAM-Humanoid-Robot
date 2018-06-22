"""""""""""""""""""""""""""""""""""""""""""""
DESCRIPTION:
    This file contains the FastSLAM class for use in performing SLAM on 
        two transform systems (sensor to body, body to world).
    The FastSLAM class stores the map data and performs SLAM updates.
    The FastSLAM class assumes that the user has access to:
        1) lidar data
        3) sensor to body transforms (usually found through an IMU and robot parameters)
        4) some method of getting rough odometries
    FastSLAM makes a few approximations:
        1) the room is has a level floor
        2) the only body to world transform is a shift down by a constant amount
        3) 
    
"""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from utils.misc import *

class FastSLAM():
    def __init__(self, x_range = (-30,30), y_range = (-20,20), map_res = 0.05,
                 dtheta_res = 1, pos_sweep_ind = np.arange(-4,5,1), 
                 lidar_angles = np.arange(-135,135.25,0.25), lidar_range = (0.1,30),
                 lidar_sweep_ind = np.arange(-4,5,1), 
                 num_particles = 50, x_sigma = 0.001, y_sigma = 0.001, theta_sigma = 0.015):
        assert map_res > 0, "map_res must be positive"
        assert len(x_range) == 2, "x_range must be a tuple or list with 2 values (min,max)"
        assert len(y_range) == 2, "y_range must be a tuple or list with 2 values (min,max)"
        assert len(lidar_range) == 2, "lidar_range must be a tuple or list with 2 values (min,max)"
        assert x_range[0] < x_range[1], "The first element of x_range must be less than the second element"
        assert y_range[0] < y_range[1], "The first element of y_range must be less than the second element" 
        assert num_particles > 0, "There must be a positive number of particles"
        assert x_sigma > 0, "standard deviations must be positive"
        assert y_sigma > 0, "standard deviations must be positive"
        assert theta_sigma > 0, "standard deviations must be positive"
        
        # Particle Initializations
        self.num_particles = num_particles  #number of particles used for particle filter
        self.mu = np.zeros((num_particles,4,4)) #particles used for particle filter
        for i in range(self.num_particles):
            self.mu[i] = get_T(0,0,0,0,0,0)
        self.mu_weights = np.ones((num_particles))/num_particles #weights for each respective particle
        self.x_sigma = x_sigma          #x standard deviation
        self.y_sigma = y_sigma          #y standard deviation
        self.theta_sigma = theta_sigma  #theta(yaw) standard deviation
        self.max_ind = 0                #index of the best performing particle
        
        # Map Initializations
        self.map_res = map_res          #map resolution
        self.xmin = x_range[0]          #dimensions of map
        self.xmax = x_range[1]          #dimensions of map
        self.ymin = y_range[0]          #dimensions of map
        self.ymax = y_range[1]          #dimensions of map
        self.sizex = int(np.ceil((self.xmax - self.xmin) / self.map_res + 1)) #number of cells in x direction
        self.sizey = int(np.ceil((self.ymax - self.ymin) / self.map_res + 1)) #number of cells in y direction
        self.map = np.zeros((self.sizex,self.sizey),dtype=np.float)     #map initialization
        
        # Lidar Initialization
        self.lidar_angles = np.deg2rad(lidar_angles.T)           #list of all angles used for the lidar
        self.lmin = lidar_range[0]
        self.lmax = lidar_range[1]
        
        # Sweep Correction Initialization
        self.dxy = pos_sweep_ind*self.map_res                           #x and y values that are swept over for getting near particle correlations
        self.dtheta = np.deg2rad(lidar_sweep_ind*dtheta_res)    #theta values that are swept over for getting near particle correlations (in radians)
        self.T_theta = np.zeros((self.dtheta.shape[0],4,4))         #transformation matrix representation of the theta sweep values
        for i in range(self.dtheta.shape[0]):
            self.T_theta[i] = get_T(0,0,self.dtheta[i],0,0,0)
        
        # Initialized values used to reduce parameter reinitialization
        self._x_im = np.linspace(self.xmin,self.xmax,self.sizex)    #indices in map (calculated once here so that it is not calculated again)
        self._y_im = np.linspace(self.ymin,self.ymax,self.sizey)    #indices in map (calculated once here so that it is not calculated again)


    # FUNCTION: map_correlation
    # INPUT 
    # thresh_map        the thresholded map 
    # vp(0:2,:)         occupied x,y positions from range sensor (in physical unit) 
    #
    # OUTPUT 
    # cpr               matrix of sum of the cell values of all the positions hit by range sensor (one for each xy sweep combination)
    def map_correlation(self, thresh_map, vp):
        # get the correlation matrix across the x and y sweeps for each vector position, vp
        cpr = np.zeros((self.dxy.shape[0], self.dxy.shape[0]))
        for jy in range(0,self.dxy.size):
            y1 = vp[1,:] + self.dxy[jy] # 1 x 1076
            iy = np.int16(np.round((y1-self.ymin)/self.map_res))
            for jx in range(0,self.dxy.size):
                x1 = vp[0,:] + self.dxy[jx] # 1 x 1076
                ix = np.int16(np.round((x1-self.xmin)/self.map_res))
                valid = np.logical_and(np.logical_and((iy >=0), (iy < self.sizey)), \
                                       np.logical_and((ix >=0), (ix < self.sizex)))
                cpr[jx,jy] = np.sum(thresh_map[ix[valid],iy[valid]])
        return cpr
    
    
    # FUNCTION: update_map
    # INPUT 
    # z                 lidar measurements   
    # T_w2b             body to world transform matrix
    # T_b2l             lidar to body transform matrix
    #
    # OUTPUT 
    # self.map          updated map in self.map (not returned)
    def update_map(self, z,T_w2b,T_b2l):
        # get valid lidar measurements and corresponding angles associated with them
        indValid = np.logical_and((z < self.lmax),(z > self.lmin))
        ranges = z[indValid]
        angles = self.lidar_angles[indValid]
        
        # format the xy position of lidar measurements in the lidar frame as a position vector
        xs0 = np.array([ranges*np.cos(angles)])
        ys0 = np.array([ranges*np.sin(angles)])
        z_temp0 = np.zeros(xs0.shape)
        z_temp1 = np.ones(xs0.shape)
        p_s = np.vstack((xs0[0],ys0[0],z_temp0[0],z_temp1))
        
        # convert position vector of lidar measurements from the lidar frame into the world frame
        p_w = np.empty(p_s.shape)
        for i in range(p_s.shape[1]):
            p_w[:,i] = np.matmul(np.matmul(T_w2b,T_b2l),p_s[:,i])
        
        # get associated map indicies of lidar measurements in the world frame
        xis = np.ceil((p_w[0,:] - self.xmin) / self.map_res).astype(np.int16)-1
        yis = np.ceil((p_w[1,:] - self.ymin) / self.map_res).astype(np.int16)-1
        
        # get current robot xy position in the world frame
        curr_x = T_w2b[0,3]
        curr_y = T_w2b[1,3]
        
        # get associated map indicies of current robot xy position in the world frame
        ind_curr_x = np.ceil((curr_x - self.xmin) / self.map_res).astype(np.int16)-1
        ind_curr_y = np.ceil((curr_y - self.ymin) / self.map_res).astype(np.int16)-1
    
        # for each lidar measurement
        for i in range(xis.shape[0]):
            # create a ray from current position to the lidar position
            (x_test,y_test) = bresenham2D(ind_curr_x, ind_curr_y, xis[i], yis[i])
            
            # make sure that the ray lies within the bounds of the map (ignore if it does not) ---to be improved
            rayGood = np.logical_and(np.logical_and(np.logical_and((x_test > 1), (y_test > 1)), (x_test < self.sizex)), (y_test < self.sizey))
            x_test = x_test[rayGood].astype(np.int16)
            y_test = y_test[rayGood].astype(np.int16)
            
            # if the ray has more than one value (condition is just a safety net and should not have to be used)
            # add log(9) to the end point of the ray
            # subtract log(9) to each intermitent point on the ray
            if len(x_test > 1):
                self.map[x_test[0:-1],y_test[0:-1]] -= np.log(9.)
                self.map[x_test[-1],y_test[-1]] += np.log(9.)
                
                
    # FUNCTION: predict
    # INPUT 
    # T_w2b_new         new body to world transform matrix from lidar odometry
    # T_w2b_last        last body to world transform matrix from lidar odometry
    # OUTPUT 
    # mu_next           predicted particles
    def predict(self, T_w2b_last, T_w2b_new):
        # sample noise for particles
        noise_theta = np.random.normal(0,self.theta_sigma,(self.num_particles))
        noise_x = np.random.normal(0,self.x_sigma,(self.num_particles))
        noise_y = np.random.normal(0,self.y_sigma,(self.num_particles))
        
        # get dead reckoning odomentry
        odo_dr = np.matmul(np.linalg.inv(T_w2b_last),T_w2b_new)
        
        # predict the next particle location for each particle
        for i in range(self.num_particles):        
            N = get_T(0,0,noise_theta[i],noise_x[i],noise_y[i],0)
            Pred = np.matmul(odo_dr,N)  
            self.mu[i] = np.matmul(self.mu[i],Pred)
        #return particles for visualization
        return self.mu[self.max_ind]


    # FUNCTION: update
    # INPUT 
    # z                 lidar measurements
    # T_b2l             lidar to body transformation matrix
    # floor_dist        distance from body to floor
    # OUTPUT 
    # mu_next           particles corrected for using map correlation (not returned)
    # mu_best           particle with the highest correlation
    def update(self,z,T_b2l,floor_dist): 
        # format the xy position of lidar measurements in the lidar frame as a position vector
        indValid = np.logical_and((z < self.lmax),(z > self.lmin))
        ranges = z[indValid]
        angles = self.lidar_angles[indValid]
        
        # format the xy position of lidar measurements in the lidar frame as a position vector
        xs0 = np.array([ranges*np.cos(angles)])
        ys0 = np.array([ranges*np.sin(angles)])
        z_temp0 = np.zeros(xs0.shape)
        z_temp1 = np.ones(xs0.shape)
        p_s = np.vstack((xs0[0],ys0[0],z_temp0[0],z_temp1)).T
        
        # convert position vector of lidar measurements from the lidar frame into the body frame
        p_b = np.zeros(p_s.shape)
        for i in range(p_s.shape[0]):
            p_b[i] = np.matmul(T_b2l,p_s[i])
        
        # Ignore values that hit the floor
        p_b[:,2] = p_b[:,2] + floor_dist
        indNotFloor = (p_b[:,2] > 0.1).T
        p_b = p_b[indNotFloor]
        
        # convert position vector of lidar measurements from the body frame into the world frame
        p_w = np.zeros((self.dtheta.shape[0],self.num_particles,p_b.shape[0],4))
        for i in range(self.num_particles):
            for j in range(p_b.shape[0]):
                for k in range(self.dtheta.shape[0]):
                    p_w[k,i,j] = np.matmul(self.T_theta[k],np.matmul(self.mu[i],p_b[j]))
    
        # get position values for easier access (theta_sweep, particle,lidar_points,position)
        v_w = p_w[:,:,:,0:2]
        
        # get the thresholded map
        thresh_map = np.zeros(self.map.shape).astype(np.uint16)
        thresh = self.map > 0.
        thresh_map[thresh] = 1
        
        # setup for loop
        max_val = np.zeros(self.num_particles)
        map_corr = np.zeros((self.T_theta.shape[0],self.dxy.shape[0],self.dxy.shape[0]))
        for i in range(self.num_particles):
            #get the correlation for each permutation of dx,dy, and dtheta
            for j in range(self.T_theta.shape[0]):
                map_corr[j] = self.map_correlation(thresh_map,v_w[j,i].T)
            #find the location around the particle with best correlation
            #-------------to be optimized--------------------
            max_val[i] = np.amax(map_corr)
            if not np.allclose(map_corr,0): #check to avoid float math errors
                temp_theta = np.where(map_corr == np.amax(map_corr))[0][0]
                temp_x = np.where(map_corr == np.amax(map_corr))[1][0]
                temp_y = np.where(map_corr == np.amax(map_corr))[2][0]
            #-------------to be optimized--------------------
                T_new = get_T(0,0,self.dtheta[temp_theta],self.dxy[temp_x],self.dxy[temp_y],0)
                self.mu[i] = np.matmul(T_new,self.mu[i])
        # weight each pixel based on how well they correlate with the map
        max_val = soft_max(max_val)
        self.mu_weights = (self.mu_weights*max_val)/np.sum(self.mu_weights*max_val)
        
        #get best performing particle
        self.max_ind = np.where(self.mu_weights == np.amax(self.mu_weights))[0][0]
        mu_best = self.mu[self.max_ind] #Note that the best particle is also the body to world transform
        #update the map based on the best performing particle
        return mu_best
    
    
    # FUNCTION: get_N_eff
    # INPUT 
    #
    # OUTPUT 
    # N_eff         effective number of partices (used to determine if resampling is necessary)
    def get_N_eff(self):
        N_eff = 1/np.sum(np.exp2(self.mu_weights))
        return N_eff


    # FUNCTION: resample (stratified resampling)
    # INPUT 
    #
    # OUTPUT 
    # resampled particles and weights(not returned)
    def resample(self):
        mu_out = np.empty(self.mu.shape)
        mu_weights_out = np.ones(self.mu_weights.shape)/self.num_particles
        j = 0
        c = self.mu_weights[0]
        for i in range(self.num_particles):
            u = np.random.uniform(0,1/self.num_particles)
            beta = u+i/self.num_particles
            while beta > c:
                j = j+1
                c = c + self.mu_weights[j]
            mu_out[i] = self.mu[j]
        self.mu = mu_out
        self.mu_weights = mu_weights_out
