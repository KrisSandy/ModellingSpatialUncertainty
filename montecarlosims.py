import camera
from camerastats import Intrinsics
import camerautils as cu
from incidentangle import IncidentAngle
import numpy as np
import math

# -------------------------------------------------------------------------

# Author : Krishna Sandeep

# This class provides methods to perform Monte Carlo simulations for
# 1) Incident Angle vs Radius
# 2) Ground Plane Projections

# -------------------------------------------------------------------------

class MonteCarloSimulations():

    calib_file = 'data/calib.xml'
    intrinsics_file = 'data/intrinsic_data.csv'


    def __init__(self, camera_type='FV', width=1280, height=966):
        # initialize default parameters
        self.__camera_type = camera_type
        self.__width = width
        self.__height = height
        self.__intrinsics = Intrinsics()
        self.__intrinsics.fit_kde_6d()


    # Runs Monte Cralo simulations for ground plane projections
    # This method uses the camera module and is not included as part of the project.
    # This module can be replaced with a custom camera module and simulations can be performed   

    def run_ground_plane_sims(self, iterations=1000, grid_size=6, max_rot_x=60, model='kde', savefile=None):

        if model not in ['kde', 'gaussian']:
            raise Exception('Invalid model parameter')

        # create a fisheye camera object using the camera module 
        camera_obj = camera.CameraFisheye()
        # get the intrinsic and extrinsic parameters from the calibration xml file
        intr, extr = camera.xml_params(self.calib_file, self.__camera_type)
        # set the extrinsic rotation in Z1 direction to 90 degrees and Z2 direction to 0 degrees
        extr.rot.z1_rad = math.radians(90)
        extr.rot.z2_rad = math.radians(0)

        # create pixel grid and camera orientation at which simulations are run
        raw_x = cu.create_grid(self.__width, grid_size)
        raw_y = cu.create_grid(self.__height, grid_size)
        xrads = np.deg2rad(np.linspace(0, max_rot_x, max_rot_x, dtype=np.int16))
        n = len(raw_x)
        KEY = np.zeros([n*n*max_rot_x, 3])
        c = 0
        for i in range(n):
            for j in range(n):
                for k in range(max_rot_x):
                    KEY[c] = [raw_x[i], raw_y[j], xrads[k]]
                    c += 1

        GP = np.zeros((KEY.shape[0], 2, iterations))

        # Get the random samples from multivariate gaussian if the model is gaussian
        if model == 'gaussian':
            randomsamples = self.__intrinsics.fit_multivariate_gaussian()

        print('Running Monte Carlo Simulations for {} iterations'.format(iterations))

        # run the Monte Carlo simulations
        for i in range(iterations):
            # get a sample intrinsic parameters from the fitted 6D KDE
            if model == 'gaussian':
                intrv = randomsamples[i]
            else:
                intrv = self.__intrinsics.get_variate_6d()

            # configure the fisheye camera object with the sampled intrinsic parameters
            intr.fisheyeCoeff1 = float(intrv[0])
            intr.fisheyeCoeff2 = float(intrv[1])
            intr.fisheyeCoeff3 = float(intrv[2])
            intr.fisheyeCoeff4 = float(intrv[3])
            intr.codXoffset_px = float(intrv[4])
            intr.codYoffset_px = float(intrv[5])

            # at every grid location and camera orientation, calculate the ground projections
            # and save them in numpy array
            for k in range(KEY.shape[0]):
                px, py, xrad = KEY[k]
                extr.rot.x_rad = xrad
                camera_obj.configure(intr, extr)
                p = camera_obj.rawToGroundPlanePt([px, py])
                GP[k][0][i] = p[0]
                GP[k][1][i] = p[1]

        # save the simulations
        if savefile is None:
            np.savez('data/simulations/mc_gpn_' + model + '_' + self.__camera_type + '.npz', gp=GP, key=KEY)
        else:
            np.savez(savefile + '.npz', gp=GP, key=KEY)
        print('Saved Ground Plane simulations')
        

    # Runs Monte Cralo simulations for incident angle vs radius.
    # This method does not use the camera module.

    def run_incident_angle_sims(self, iterations=10000, max_angle=110, step_angle=1, savefile=None):

        # create a grid of thetas 
        thetas = np.around(np.arange(0, max_angle, step_angle), decimals=2)
        r = np.zeros((len(thetas), iterations))

        print('Running Monte Carlo Simulations for {} iterations'.format(iterations))

        # run the Monte Carlo simulations
        for i in range(iterations):
            # get a sample intrinsic parameters from the fitted 6D KDE
            intrv = self.__intrinsics.get_variate_6d()

            # at every theta, calculate the radius
            for t, theta in enumerate(thetas):
                r[t][i] = cu.get_radius(theta, intrv[0], intrv[1], intrv[2], intrv[3])
      
        # save the simulations
        if savefile == None:
            np.savez('data/simulations/mc_ia.npz', theta=thetas, r=r)
        else:
            np.savez(savefile + '.npz', theta=thetas, r=r)
        print('Saved Incident Angle simulations')


if __name__ == "__main__":
    sims = MonteCarloSimulations()
    # run Monte Carlo simulations for incident angle vs radius
    sims.run_incident_angle_sims()
    # run Monte Carlo simulations for ground plane projections using KDE model 
    sims.run_ground_plane_sims(max_rot_x=90)
    # run Monte Carlo simulations for ground plane projections using gaussian model 
    sims.run_ground_plane_sims(max_rot_x=90, model='gaussian')