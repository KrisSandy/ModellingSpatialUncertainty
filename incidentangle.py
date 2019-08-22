import camerautils as cu
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, fsolve
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# -------------------------------------------------------------------------

# Author : Krishna Sandeep

# This class provides functions for calculating the metrics and generating 
# visualizations using the Incident Angle vs Radius simulations

# -------------------------------------------------------------------------

class IncidentAngle():

    # initialize default parameters
    __r = None
    __thetas = None
    __cache = None
    __fit = None
    __gaufit = None

    def __init__(self):
        # initialize default parameters
        sns.set(font_scale = 1.5)


    # fit a KDE at each cluster of simulation points 
    def fit_kde(self):

        if self.__r is None:
            raise Exception('Run Monte Carlo Simulations first!!')

        # initialize default parameters
        n_points = 200
        n = self.__r.shape[0]
        X, Y, Z, Zc = np.zeros((4, n, n_points))
        U, S = np.zeros((2, n))
        Z[0] = Zc[0] = 1

        for i in range(1, n):
            # get the calculated radius at an incident angle
            data = self.__r[i]
            # fit kde
            kde = stats.gaussian_kde(data)
            # calculate mean, covariance and data points for visualization
            U[i] = np.mean(data)
            cov = np.ravel(kde.covariance)
            fac = kde.factor
            S[i] = math.sqrt(cov/fac)
            Y[i] = np.linspace(data.min(), data.max(), n_points)
            X[i] = self.__thetas[i] 
            Zt = kde(Y[i])
            # using min/max normalization for visualization purpose
            Zc[i] = (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))
            # using log function to reduce the scale of probability density
            Z[i] = np.log(Zt)

        self.__cache = {'X': X, 'Y': Y, 'U': U, 'Z': Z, 'Zc': Zc, 'S': S}
        self.__fit = 'kde'


    # fit a KDE at each cluster of simulation points 
    def fit_gaussuian_with_ci(self, ci=95):

        if self.__r is None:
            raise Exception('Run Monte Carlo Simulations first!!')

        # initialize default parameters
        n_points = 200
        n = self.__r.shape[0]
        X, Y, Z, Zc = np.zeros((4, n, n_points))
        U, S = np.zeros((2, n))
        Z[0] = Zc[0] = 1

        for i in range(1, n):
            # get the calculated radius at an incident angle
            data = self.__r[i]
            # calculate mean and standard deviation of the data
            U[i], S[i] = stats.norm.fit(data)
            # generate data points for visualization
            Y[i] = np.linspace(data.min(), data.max(), n_points)
            X[i] = self.__thetas[i] 
            Zt = stats.norm.pdf(Y[i], U[i], S[i])
            # using min/max normalization for visualization purpose
            Zc[i] = (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))
            # using log function to reduce the scale of probability density
            Z[i] = np.log(Zt)

        # Below code calculates the mean, min limit and max limit at given CI level.
        if ci == 90:
            alpha=1.64
        elif ci == 99:
            alpha = 2.58
        else:
            alpha = 1.96
        Ulow = U - (alpha * S)
        Uupp = U + (alpha * S)

        # fit the curve to calculate the fisheye coefficient at mean, min and max levels
        fitvals_mean, _ = curve_fit(cu.get_radius, X[:, 0], U)
        fitvals_min, _ = curve_fit(cu.get_radius, X[:, 0], Ulow)
        fitvals_max, _ = curve_fit(cu.get_radius, X[:, 0], Uupp)

        self.__gaufit = np.array([fitvals_min, fitvals_mean, fitvals_max])
        self.__cache = {'X': X, 'Y': Y, 'U': U, 'Ulow': Ulow, 'Uupp': Uupp, 'Z': Z, 'Zc': Zc, 'S': S}
        self.__fit = 'gau'

    
    # Visualize the monte carlo simulations - generates scatter plot of Incident angle vs Radius
    def visualize_simulations(self):
        if self.__r is None:
            raise Exception('Run Monte Carlo Simulations first!!')

        plt.scatter(self.__thetas.repeat(self.__r.shape[1]).reshape(self.__r.shape), self.__r, alpha=0.5)
        plt.xlabel('Incident Angle (degrees)')
        plt.ylabel('Radius (pixels)')
        plt.title('Monte Carlo Simulation - Incident Angle vs Radius Scatter plot')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.show()


    # Visualize fitted simulations - generates 3D surface / wireframe / 2D density plot of Incident angle vs Radius
    # This method also produces pdf plots at a given incident angle
    def visualize_fit(self, type='density', theta=60):
        if self.__cache is None:
            raise Exception('Fit the data first!!')

        if type == 'density':
            plt.pcolormesh(self.__cache['X'], self.__cache['Y'], self.__cache['Zc'], cmap=plt.cm.coolwarm)
            plt.xlabel('Incident Angle (degrees)')
            plt.ylabel('Radius (pixels)')
            plt.ticklabel_format(useOffset=False, style='plain')
            plt.title('Monte Carlo Simulation - Incident Angle vs Radius Density plot')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Min-Max Normalised')
            plt.show()
        elif type == 'surface' or type == 'wireframe':
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if type == 'surface':
                ax.plot_surface(self.__cache['X'], self.__cache['Y'], self.__cache['Z'], rstride=5, cstride=5, alpha=0.5, \
                                facecolors=plt.cm.viridis(self.__cache['Zc']))
                ax.contour(self.__cache['X'], self.__cache['Y'], self.__cache['Z'], zdir='x')
            else:
                ax.plot_wireframe(self.__cache['X'], self.__cache['Y'], self.__cache['Z'], rstride=5, cstride=5, cmap='coolwarm')
            ax.set_xlabel('Incident Angle (degrees)', labelpad=20)
            ax.set_ylabel('Radius (pixels)', labelpad=20)
            ax.set_zlabel('log(density)')
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.view_init(azim=-60, elev=-154)
            ax.set_title('Monte Carlo Simulation - Distribution of radius with respect to incident angle')
            plt.show()
        elif type == 'single':
            data = self.__r[theta]
            # produce data points for kde plot
            x_grid = np.linspace(data.min(), data.max(), 1000)
            kde = stats.gaussian_kde(data)
            Y_kde = kde(x_grid)
            plt.hist(data, 'auto', fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
            plt.plot(self.__cache['Y'][theta], np.exp(self.__cache['Z'][theta]), label='Gaussian')
            plt.plot(x_grid, Y_kde, label='KDE')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Histogram, Gaussian, and KDE plot of radius at an incident angle '+str(theta) +' degrees')
            plt.legend(loc='upper right')
            plt.show()            
        else:
            raise Exception('unknown type parameter')

    
    # This method loads the Monte Carlo simulations from the saved file
    def load_simulations(self, file='data/simulations/mc_ia'):
        container = np.load(file + '.npz')
        self.__r = container['r']
        self.__thetas = container['theta']
        print('Loaded {} simulations'.format(self.__r.shape[1]))


    # This method saves the fitted simulations, a table with incident angle , mean of radius, and standard deviation
    def save_fitted_data(self):
        if self.__cache is None:
            raise Exception('Fit the data first!!')

        data = np.stack([self.__cache['X'][:, 0], self.__cache['U'], self.__cache['S']], axis=-1)
        df = pd.DataFrame(data, columns=['angle', 'radius_mean', 'radius_sd'])
        df.to_csv('results/ia_' + self.__fit + 'xref.csv', index=False)
        np.savez('data/ia_gaufit.npz', gaufit=self.__gaufit)
        print('Saved {} fitted data'.format(self.__fit))

    # This method returns the fisheye coefficients of min, mean and max curves of Incident angle and radius
    def get_ci_parms(self, ci=95):
        if self.__gaufit is None:
            raise Exception('Fit the data first or Load fitted data!!')

        return self.__gaufit

    # This method loads the fisheye coefficients of min, mean and max curves of Incident angle and radius
    def load_fitted_gaudata(self):
        container = np.load('data/ia_gaufit.npz')
        self.__gaufit = container['gaufit']


    # This method generates the visualizations for the gaussian fitted curves
    def visualize_ci_curve(self):
        if self.__cache is None:
            raise Exception('Fit the data first!!')

        if self.__fit != 'gau':
            raise Exception('Perform gaussian fit first!!!')


        plt.plot(self.__cache['X'][:, 0], self.__cache['U'], label='mean')
        plt.plot(self.__cache['X'][:, 0], self.__cache['Ulow'], label='95% CI lower limit')
        plt.plot(self.__cache['X'][:, 0], self.__cache['Uupp'], label='95% Ci upper limit')
        plt.xlabel('Incident Angle (degrees)')
        plt.ylabel('Radius (pixels)')
        plt.title(r'95% CI fitted curve - Incident Angle vs Radius')
        plt.legend(loc='lower right')
        plt.show()


if __name__ == "__main__":
    ia = IncidentAngle()
    ia.load_simulations()
    ia.visualize_simulations()
    ia.fit_gaussuian_with_ci()
    ia.visualize_fit(type='density')
    ia.visualize_fit(type='surface')
    ia.save_fitted_data()
    ia.visualize_ci_curve()