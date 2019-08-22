import numpy as np 
from xml.dom import minidom

# -------------------------------------------------------------------------

# Author : Krishna Sandeep

# This library provides helper functions for various camera operations.

# -------------------------------------------------------------------------

# Class to define the extrinsic parameters of the camera
class extr:
    rot_x_deg = None
    rot_z1_deg = None
    rot_z2_deg = None
    pointx_mm = None
    pointy_mm = None
    pointz_mm = None


def get_radius(t, k1, k2, k3, k4):

    # Radius calculated using below fourth order polynomial
    # r = k1*theta + k2*theta^2 + k3*theta^3 + k4*theta^4

    # convert angle to radians
    t_r = np.deg2rad(t)

    # return radius in pixels
    return k1*t_r + k2*t_r**2 + k3*t_r**3 + k4*t_r**4


def create_grid(n, s):
    # returns a pixel grid with center pixel always included
    return np.unique(np.concatenate([np.linspace(0, n/2, s, dtype=np.int16), np.linspace(n/2, n, s, dtype=np.int16)]))


def get_camera_specs(calib_file = 'data/calib.xml', camera_type='FV'):
    # This method reads the xml calibration file and returns the extrinsic parameters object.

    # read the calibration xml file
    calib = minidom.parse(calib_file)
    calib_cameras = calib.getElementsByTagName('camera')

    # create empty extrinsics object
    ext = extr()
    for calib_camera in calib_cameras:
        if calib_camera.attributes['name'].value == camera_type:
            # populate the extrinsics object with the data from the calibration file
            ext.pointx_mm = float(calib_camera.getElementsByTagName('pointx_mm')[0].attributes['value'].value)
            ext.pointy_mm = float(calib_camera.getElementsByTagName('pointy_mm')[0].attributes['value'].value)
            ext.pointz_mm = float(calib_camera.getElementsByTagName('pointz_mm')[0].attributes['value'].value)
            ext.rot_x_deg = float(calib_camera.getElementsByTagName('rot__x_deg')[0].attributes['value'].value)
            ext.rot_z1_deg = float(calib_camera.getElementsByTagName('rot_z1_deg')[0].attributes['value'].value)
            ext.rot_z2_deg = float(calib_camera.getElementsByTagName('rot_z2_deg')[0].attributes['value'].value)
    
    return ext
