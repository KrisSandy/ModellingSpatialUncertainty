import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import seaborn as sns

# -------------------------------------------------------------------------

# Author : Krishna Sandeep

# This class provides functions to perform 
# 1) Normality test on the intrinsic parameters data
# 2) Fit the individual intrinsic parameters data with KDE
# 3) Fit all the intrinsic parameters with KDE  
# 4) Print the covariance matrix of the fitted 6D KDE
# 5) Fit all the intrinsic parameters with Multivariate Gaussian

# -------------------------------------------------------------------------


class Intrinsics():

    __kde_1d_functions = list()
    __kde_2d = None
    __kde_6d = None
    __kde_4d = None

    def __init__(self, file='data/intrinsic_data.csv'):
        self.data = pd.read_csv(file)


    def test_for_normality(self, alpha=0.05):

        # Notmality test on the intrinsic parameters data

        for col in self.data.columns:
            x = self.data.loc[:,col].values
            # Use scipy normaltest function to perform normality test 
            stat, p = stats.normaltest(x)
            print('stats : {}, p : {}'.format(stat, p))
            # if p value is less than significance level, then the distribution is gaussian
            # otherwise the distribution is not gaussian
            if p < alpha:
                print('{} - gaussian'.format(col))
            else:
                print('{} - not gaussian'.format(col))


    def print_stats(self):
        # describe the data frame
        print(self.data.describe())


    def __get_1d_grid(self, x, n=1000):
        # create 1D grid
        return np.linspace(x.min(), x.max(), n)

    def __get_kde_1d(self, x):
        # fit kde using the data passed in the parameter
        kde = stats.gaussian_kde(x)
        x_grid = self.__get_1d_grid(x)
        # return the fitted kde, 1D grid and the sampled data points produced using the 1D grid and KDE
        return kde, x_grid, kde(x_grid)


    def fit_kde_1d(self):

        # For the individual intrinsic parameters fir the kde and plot the sampled data from fitted kde

        fig, ax = plt.subplots(3, 2, sharey=True)
        colnames = ['Fisheye Coefficient K1', 'Fisheye Coefficient K2', 'Fisheye Coefficient K3', 'Fisheye Coefficient K4', 'Center of Distortion Offset X', 'Center of Distortion Offset Y']

        for i, col in enumerate(self.data.columns):
            p = self.data.loc[:,col].values
            
            p_kde, p_grid, pdf = self.__get_kde_1d(p)

            self.__kde_1d_functions.append(p_kde)

            r = i//2
            c = i%2
            ax[r][c].plot(p_grid, pdf, linewidth=3)
            ax[r][c].set_xlim(p.min(), p.max())
            ax[r][c].set_title(colnames[i])
            ax[r][c].set_xlabel('value')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1,
        wspace=0.35)
        plt.show()


    def fit_kde_6d(self):
        # Fit a 6D KDE using all the intrinsic parameters data
        self.__kde_6d = stats.gaussian_kde(self.data.values.T)

    
    def get_variate_6d(self, size=1):
        # return sampled intrinsic parameters (6 parameters) data from fitted kde
        return self.__kde_6d.resample(size=size)

    
    def get_covariance_6d(self):
        # return the covariance matrix of the fitted KDE
        return self.__kde_6d.covariance


    def get_factor_6d(self):
        # return the multiplication factor for the covariance matrix of fitted KDE
        return self.__kde_6d.factor


    def fit_multivariate_gaussian(self, samples=1000):
        # Fit all the intrinsic parameters data using the Multivariate Gaussian distribution
        data = self.data.values
        # calculate Mean and covariance matrix for the intrinsic parameter data
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        # get the samples from the fitted distribution 
        randomsamples = np.random.multivariate_normal(mean,cov,samples)
        return randomsamples


if __name__ == '__main__':
    intrinsics = Intrinsics()
    intrinsics.fit_multivariate_gaussian()
    intrinsics.test_for_normality()
    intrinsics.fit_kde_1d()
    np.set_printoptions(suppress=True)
    print(intrinsics.get_covariance_6d())
