import camerautils as cu
from IncidentAngle import IncidentAngle
import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import griddata
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches
import sys
import os

# -------------------------------------------------------------------------

# Author : Krishna Sandeep

# This class provides functions for calculating the metrics and generating 
# visualizations using the Ground plane projection simulations

# -------------------------------------------------------------------------

class GroundPlane():

    # initialize default parameters
    calib_file = 'data/calib.xml'
    __KEY = None
    __GP = None
    __GPDF = None
    __model = None


    def __init__(self, camera_type='FV', width=1280, height=966):
        # initialize default parameters
        self.__camera_type = camera_type
        self.__width = width
        self.__height = height
        sns.set(font_scale = 1.5)


    # Helper function to get the incident angle given radius.
    # This method uses the incident angle to get the fisheye coefficients of min, mean and max curves of Incident angle and radius
    # and solves the equation to find the incident angle
    def __get_theta(self, px, py):
        # create an object for incident angle and get the coefficients
        ia = IncidentAngle()
        ia.load_fitted_gaudata()
        coef_min, coef_mean, coef_max = ia.get_ci_parms() 
        # calculate the center pixel
        center_px = [self.__width/2, self.__height/2] 
        # definition of the camera equation 
        camera_eq = lambda t, r, k1, k2, k3, k4: r - (k1*t + k2*t**2 + k3*t**3 + k4*t**4)
        # calculate the radius
        r = distance.euclidean([px, py], center_px)
        # solve for mean, min and max theta values
        tmean = np.rad2deg(fsolve(camera_eq, 0, args=(r, coef_mean[0], coef_mean[1], coef_mean[2], coef_mean[3])))[0]
        tmin = np.rad2deg(fsolve(camera_eq, 0, args=(r, coef_min[0], coef_min[1], coef_min[2], coef_min[3])))[0]
        tmax = np.rad2deg(fsolve(camera_eq, 0, args=(r, coef_max[0], coef_max[1], coef_max[2], coef_max[3])))[0]
        return (tmin, tmax, tmean) 


    # Below function calculated the metrics for the cluster of ground projections
    def __fit_single_cluster(self, px, py, data, returngrid=False):
        n = 100j
        # get the camera extrinsics
        extr = cu.get_camera_specs(self.calib_file, self.__camera_type)
        camera_extr = np.array([extr.pointx_mm, extr.pointy_mm, extr.pointz_mm])
        
        fitinfo = dict()

        # calculate the mean
        fitinfo['mean'] = np.mean(data, axis=1)
        # check if the metrix is singular
        if np.linalg.cond(data) < 1/sys.float_info.epsilon:
            # fit a KDE
            kde = stats.gaussian_kde(data)
            # get the covariance matrix
            fitinfo['cov'] = kde.covariance/kde.factor
            # get the eigenvalues and eigenvectors
            fitinfo['w'], fitinfo['v'] = np.linalg.eig(fitinfo['cov'])
            # calculate the major and minor axis of the ellipse by using 95% CI of xhi squared distribution
            fitinfo['a'], fitinfo['b'] = np.sqrt(fitinfo['w']*5.991)
            # calculate data points for visualizations
            if returngrid:
                fitinfo['X'], fitinfo['Y'] = np.mgrid[data[0].min():data[0].max():n, 
                                            data[1].min():data[1].max():n]
                xy = np.vstack([fitinfo['X'].ravel(), fitinfo['Y'].ravel()])
                Z = kde(xy)
                fitinfo['Z'] = np.reshape(Z, fitinfo['X'].shape)
            # calculate the ground mean
            ground_mean = np.array([fitinfo['mean'][0], fitinfo['mean'][1], 0])
            up = ground_mean - camera_extr
            vp = np.array([0, 0, 0]) - ground_mean
            # calculate projection angle as the angle between projection line and ground plane
            fitinfo['alpha'] = 180 - np.degrees(np.arccos(np.dot(up, vp) / (np.linalg.norm(up) * np.linalg.norm(vp))))
            # calculate the distance 
            fitinfo['d'] = distance.euclidean(camera_extr, ground_mean) 
            return fitinfo
        return None


    # Below method visualizes the simulations at a specified camera orientation
    def visualize_simulations(self, xdeg=60):
        if self.__GP is None:
            raise Exception('Run Monte Carlo Simulations first!!')       

        filt = np.where(np.rad2deg(self.__KEY[:,2]).astype(int) == xdeg)

        for k in filt[0]:
            plt.scatter(self.__GP[k][0], self.__GP[k][1])
        plt.show()


    # Below method generates the visualization of monte carlo simulation with respect to car,
    # scatter plot pf the cluster and KDE of the custer.
    def save_scatters_1p(self, px=640, py=483, save=True):
        if self.__GP is None:
            raise Exception('Run Monte Carlo Simulations first!!')

        # get index of the data points matching the pixel location and camera orientation
        idxs = np.where((self.__KEY[:,0] == px) & (self.__KEY[:,1] == py))[0]

        # get the camera locations (extrinsic parameters)
        eMVL = cu.get_camera_specs(self.calib_file, 'MVL')
        eMVR = cu.get_camera_specs(self.calib_file, 'MVR')
        eRV = cu.get_camera_specs(self.calib_file, 'RV')
        eFV = cu.get_camera_specs(self.calib_file, 'FV')   
        
        tmin, tmean, tmax = self.__get_theta(px, py)

        plots_folder = 'plots/Ground Plane/simplots/{}_{}'.format(px, py)
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Generate visualizations for each index
        for i in idxs:
            px, py, xrad = self.__KEY[i]
            # get the data from the index 
            data = np.vstack([self.__GP[i, 0, :], self.__GP[i, 1, :]])

            # calculate the metrics
            fitinfo = self.__fit_single_cluster(px, py, data, returngrid=True)

            if fitinfo is None:
                continue

            # generate visualizations of monte carlo simulation with respect to car
            semis = np.sqrt(fitinfo['w'])

            fig = plt.figure(figsize=(16, 12))

            ax = fig.add_subplot(2, 1, 1)
            ax.scatter(data[0], data[1], alpha=0.5)
            plt.plot(eFV.pointx_mm, eFV.pointy_mm, color='forestgreen', marker='o')
            plt.plot(eRV.pointx_mm, eRV.pointy_mm, color='forestgreen', marker='o')
            plt.plot(eMVL.pointx_mm, eMVL.pointy_mm, color='forestgreen', marker='o')
            ax.plot(eMVR.pointx_mm, eMVR.pointy_mm, color='forestgreen', marker='o')
            rec = patches.Rectangle([eFV.pointx_mm, eMVL.pointy_mm], abs(eFV.pointx_mm)+abs(eRV.pointx_mm), abs(eMVR.pointy_mm)+abs(eMVL.pointy_mm), alpha=0.2, color='gray')
            ax.add_patch(rec)
            ax.set_xlim([-3200, 4500])
            ax.set_ylim([-6000, 6000])
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')              

            # generate visualizations of KDE of the custer
            ax = fig.add_subplot(2, 2, 3, projection='3d')
            ax.plot_surface(fitinfo['X'], fitinfo['Y'], fitinfo['Z'], rstride=3, cstride=3, cmap=plt.cm.viridis, linewidth=0.0, alpha=0.75)
            ax.view_init(azim=-96, elev=7)
            ax.set_xlim([fitinfo['mean'][0]-60, fitinfo['mean'][0]+60])
            ax.set_ylim([fitinfo['mean'][1]-50, fitinfo['mean'][1]+50])
            ax.set_zlim([0, 0.005])
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('pd')


            # generate visualizations of scatter plot pf the cluster
            ax = fig.add_subplot(2, 2, 4)
            ax.scatter(data[0], data[1], alpha=0.2)
            ax.scatter(fitinfo['mean'][0], fitinfo['mean'][1], marker='^', color='green')
            ax.set_xlim([fitinfo['mean'][0]-60, fitinfo['mean'][0]+60])
            ax.set_ylim([fitinfo['mean'][1]-60, fitinfo['mean'][1]+60])
            ax.axis('equal')
            ax.set_aspect('equal') 
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')               
            ax.quiver(fitinfo['mean'][0], fitinfo['mean'][1], 
                fitinfo['v'][:,0]*np.sqrt(fitinfo['w'][0]*5.991), fitinfo['v'][:,1]*np.sqrt(fitinfo['w'][1]*5.991), width=0.0025)

            # create subtitle
            siminfo = list()
            siminfo.append(r'$Pixel$' + ': ({:.0f}, {:.0f})'.format(px, py))
            siminfo.append(r'$\theta$' + ': {:.0f}'.format(tmean))
            siminfo.append(r'$\delta\theta$' + ': [{:.2f}, {:.2f}]'.format(tmin, tmax))
            siminfo.append(r'$\alpha$' + ': {:.2f}'.format(fitinfo['alpha']))
            siminfo.append(r'$d$' + ': {:.2f}'.format(fitinfo['d']))
            siminfo.append(r'$a$' + ': {:.2f}'.format(max(semis)))
            siminfo.append(r'$b$' + ': {:.2f}'.format(min(semis)))
            fig.text(x=-0.1, y=2.4, s='Monte Carlo simulations of Ground Plane projections', fontsize=16, weight='bold', ha='center', va='top', transform=ax.transAxes)
            fig.text(x=-0.1, y=2.30, s=', '.join(siminfo), fontsize=13, weight='bold', alpha=0.75, ha='center', va='top', transform=ax.transAxes)

            xdeg = 90-math.degrees(xrad)
            pltname = '{:.0f}_{:.0f}_{:02.0f}.png'.format(px, py, abs(xdeg))
            if save:
                plt.savefig(plots_folder+'/'+pltname, dpi=150)
            else:
                plt.show()
            plt.close()
            print('Saved plot ',pltname)


    # Fit the simulations and create a dataframe with all the metrics calculated 
    def fit_simulations(self, d_max=5000):
        if self.__GP is None:
            raise Exception('Run Monte Carlo Simulations first!!')     

        self.__GPDF = pd.DataFrame()

        o = 0
        for k in range(self.__KEY.shape[0]):
            # get the key (pixel location and camera orientation) of the simulations
            px, py, _ = self.__KEY[k]
            # get the data for the key from the array
            data = np.vstack([self.__GP[k, 0, :], self.__GP[k, 1, :]])

            # calculate incident angle
            tmin, tmean, tmax = self.__get_theta(px, py)
            # calculate metrics
            fitsims = self.__fit_single_cluster(px, py, data)

            # add a row in the dataframe with the metrics calculated
            if fitsims is not None and fitsims['d'] < d_max:
                self.__GPDF.loc[o, 'px'] = px
                self.__GPDF.loc[o, 'py'] = py
                self.__GPDF.loc[o, 'alpha'] = fitsims['alpha']
                self.__GPDF.loc[o, 'd'] = fitsims['d']
                self.__GPDF.loc[o, 'm1'], self.__GPDF.loc[o, 'm2'] = fitsims['mean']
                self.__GPDF.loc[o, 'c1':'c4'] = np.ravel(fitsims['cov'])
                self.__GPDF.loc[o, 'a'], self.__GPDF.loc[o, 'b'] = fitsims['a'], fitsims['b']
                self.__GPDF.loc[o, 'v11'], self.__GPDF.loc[o, 'v12'] = fitsims['v'][:, 0]
                self.__GPDF.loc[o, 'v21'], self.__GPDF.loc[o, 'v22'] = fitsims['v'][:, 1]
                self.__GPDF.loc[o, 'tmean'] = tmean
                self.__GPDF.loc[o, 'tmin'] = tmin
                self.__GPDF.loc[o, 'tmax'] = tmax
                o += 1

        print('Fitting simulations complete')


    # Load the simulations
    def load_simulations(self, model='kde', simfile=None):
        if model not in ['kde', 'gaussian']:
            raise Exception('Invalid model parameter')

        if simfile is None:
            container = np.load('data/simulations/mc_gpn_' + model + '_' + self.__camera_type + '.npz')
        else:
            container = np.load(simfile + '.npz')
        self.__GP = container['gp']
        self.__KEY = container['key']
        self.__model = model
        print('Loaded {} simulations'.format(self.__GP.shape[2]))


    # Save the fitted simulations
    def save_fitted_simulations(self):
        if self.__GPDF is None:
            raise Exception('Fit Monte Carlo Simulations first!!')

        self.__GPDF.to_csv('results/gpn_' + self.__model + '_' + 'xref.csv', index=False)
        print('Saving fitted simulations')


    # load the fitted simulations for generating visualizations
    def load_fitted_simulations(self):
        self.__GPDF = pd.read_csv('results/gpn_xref.csv')
        print('Loaded fitted simulations')


    # Visualize the image grid to show the selected pixels locations in the image 
    def visualize_gridpoints(self):
        if self.__KEY is None:
            raise Exception('Run Monte Carlo Simulations first!!')

        _, ax = plt.subplots()
        grid = self.__KEY[:,0:2] 
        grid = np.unique(grid, axis=0)
        plt.scatter(grid[:,0], grid[:,1], marker='.')
        for i in range(0, grid.shape[0]):
            plt.text(grid[i, 0]+0.3, grid[i, 1]+0.3, '{:.0f}, {:.0f}'.format(grid[i, 0], grid[i, 1]), fontsize=13, weight='bold')
        plt.xlabel('width (pixels)')
        plt.ylabel('height (pixels)')
        plt.xlim(-50, 1370)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.show()


if __name__ == "__main__":
    gp = GroundPlane()
    # load the simulation run using KDE method
    gp.load_simulations()
    # produce scatter plots for pixel location 640, 483
    gp.save_scatters_1p()
    # produce scatter plots for pixel location 768, 386
    gp.save_scatters_1p(768, 386)
    # visualize the simulations at camera orientation of 60 degrees
    gp.visualize_simulations()
    # visualize the pixel locations used in the image
    gp.visualize_gridpoints()
    # fit the simulations with KDE and calculate the metrics
    gp.fit_simulations()
    # save the fitted simulations
    gp.save_fitted_simulations()
    # load the simulation run using Gaussian method
    gp.load_simulations(model='gaussian')
    # fit the simulations with KDE and calculate the metrics
    gp.fit_simulations()
    # save the fitted simulations
    gp.save_fitted_simulations()
