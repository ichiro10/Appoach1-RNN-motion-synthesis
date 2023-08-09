import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import copy
from utils.data_utils import expmap2rotmat,rotmat2expmap

def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 72

  # Structure that indicates parents for each joint
  njoints   = 23
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    #r = [ yangle, zangle, xangle ]
    r = angles[ expmapInd[i] ]
    if parent[i] == -1: # Root node
      thisRotation = expmap2rotmat(r)    
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + angles[0:3]
    else:
      thisRotation = expmap2rotmat(r)    
      #xyzStruct[i]['xyz'] = (offset[i,:]).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      #xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )
      xyzStruct[i]['xyz'] = xyzStruct[ parent[i] ]['rotation'].dot(offset[i,:]) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = xyzStruct[ parent[i] ]['rotation'].dot(thisRotation)

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  #xyz = xyz[:,[0,2,1]]

  xyz = xyz[:,[2,0,1]]
  #xyz = xyz[:,[1,0,2]]
  #xyz = xyz[:,[1,2,0]]
  #xyz = xyz[:,[2,1,0]]
  #xyz = xyz[:,[0,1,2]]

  return np.reshape( xyz, [-1] )

def revert_coordinate_space(channels, R0, T0):
  """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  # Loop through the passed posses
  for ii in range(n):
    R_diff = expmap2rotmat( channels[ii, rootRotInd] )
    R = R_diff.dot( R_prev )

    channels_rec[ii, rootRotInd] = rotmat2expmap(R)
    T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,11,12,13,14,15,12,17,18,19,12,21])
  offset = np.array([0.000000,0.000000,0.000000, -78.9148455196874, -0.18091969520497742, 0.020102189161098703, 0.0, -414.87267855035697, 0.0185271725659408, 0.0, -404.514734941311, 0.031234729879955396, 0.0, -71.92700077835221, 162.90158594902502, 78.9148455196874, -0.20274756355163961, 0.0225275079634156, 0.0, -414.872678550379, 0.018527172567007096, 0.0, -404.5147349413331, 0.031234729881856608, 0.0, -71.9270007783632, 162.90158594835202, 0.0, 96.3650515734706, -10.7072312424335, 0.0, 107.23088324950695, 0.0, 0.0, 97.88675256631294, 0.07837431205732326, 0.0, 97.77745466355071, 0.07845353876132055, 28.868820037898, 76.30747147763316, 0.0, 133.490909629952, 0.0, 0.0, 282.495855831137, 0.0, 0.0, 232.10047143838906, 0.0, 0.048038732719306594, -28.8688200378981, 76.30747147763316, 0.0, -133.490909629952, 0.0, 0.0, -282.495855831137, 0.0, 0.0, -232.10047143838895, 0.0, 0.048038732719306594, 0.0, 136.21276829949807, 0.0, 0.0, 91.0083935696261, -0.1685532350854957])
  #offset = np.array([0.0, 0.0, 0.0, -78.9148455196874, -0.18091969520497742, 0.020102189161098703, 0.0, -414.87267855035697, 0.0185271725659408, 0.0, -404.514734941311, 0.031234729879955396, 0.0, -71.92700077835221, 162.90158594902502, 78.9148455196874, -0.20274756355163961, 0.0225275079634156, 0.0, -414.872678550379, 0.018527172567007096, 0.0, -404.5147349413331, 0.031234729881856608, 0.0, -71.9270007783632, 162.90158594835202, 0.0, 96.3650515734706, -10.7072312424335, 0.0, 107.23088324950695, 0.0, 0.0, 97.88675256631294, 0.07837431205732326, 0.0, 97.77745466355071, 0.07845353876132055, 28.868820037898, 76.30747147763316, 0.0, 133.490909629952, 0.0, 0.0, 282.495855831137, 0.0, 0.0, 232.10047143838906, 0.0, 0.048038732719306594, -28.8688200378981, 76.30747147763316, 0.0, -133.490909629952, 0.0, 0.0, -282.495855831137, 0.0, 0.0, -232.10047143838895, 0.0, 0.048038732719306594, 0.0, 136.21276829949807, 0.0, 0.0, 91.0083935696261, -0.1685532350854957])

  #offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
  #offset= np.array([0.0, 0.0, 0.0, -0.07891484551968687, -0.0001809196952046932, 2.010218916126405e-05, 1.1368683772161603e-15, -0.414872678550357, 1.8527172565967477e-05, 0.0, -0.4045147349413112, 3.123472988022513e-05, 0.0, -0.07192700077835222, 0.16290158594902493, 0.07891484551968801, -0.0002027475635514975, 2.2527507963445713e-05, 0.0, -0.41487267855037885, 1.8527172567246453e-05, 1.1368683772161603e-15, -0.4045147349413331, 3.1234729881930434e-05, 0.0, -0.07192700077836324, 0.16290158594835205, 0.0, 0.09636505157347074, -0.010707231242433295, 0.0, 0.10723088324950694, 2.842170943040401e-16, 0.0, 0.09788675256631307, 7.837431205757639e-05, 0.0, 0.09777745466355058, 7.845353876177796e-05, 0.02886882003789765, 0.0763074714776333, 0.0, 0.1334909096299532, -1.4210854715202004e-16, 0.0, 0.2824958558311371, -5.684341886080802e-16, 0.0, 0.23210047143838894, 0.0, 4.8038732719533075e-05, -0.028868820037898785, 0.07630747147763352, 2.842170943040401e-16, -0.13349090962995205, -1.4210854715202004e-16, -1.4210854715202004e-16, -0.28249585583113684, 0.0, 0.0, -0.23210047143838908, -5.684341886080802e-16, 4.8038732719533075e-05, 0.0, 0.13621276829949813, 0.0, -1.1368683772161603e-15, 0.09100839356962638, -0.00016855323508551125])


  
  offset = offset.reshape(-1,3)
  
  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55],
            [59, 60, 58],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70]]
  """
  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [],
            [17, 18, 16],
            [20, 21, 19],
            [23, 24, 22],
            [],
            [26, 27, 25],
            [29, 30, 28],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [44, 45, 43],
            [],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [],
            [55, 56, 54],
            []]
  """

  expmapInd = np.split(np.arange(4,73)-1,23)

  return parent, offset, rotInd, expmapInd

