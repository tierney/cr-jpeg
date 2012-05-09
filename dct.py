#!/usr/bin/env python
import numpy
from numpy import pi,zeros
from math import cos,sqrt
from PIL import Image
from ColorSpace import ColorSpace

import numpy as np 
from scipy.fftpack import dct 

# break a single channel image into windows, ignoring the edges
def decompose_subimages(im, window_width = 8, window_height = 8 ):
    width, height,_ = im.shape
    width_intevals = width / window_width # Leverage automatic floor-ing of divided ints.
    height_intervals = height / window_height

    sub_images = []

    for height_interval in range(width_intevals):
      height_start = height_interval * window_height
      height_end = height_start + window_height
      for width_interval in range(height_intervals):
        width_start = width_interval * window_width
        width_end = width_start + window_width 
        sub_image = im[height_start:height_end, width_start:width_end, :]
        sub_images.append(sub_image)
    return sub_images


class DCT(object):
  # Example use:
  #   from dct import DCT
  #   dct = DCT('test.jpg')
  #   ret = dct.get_dcts()
  def __init__(self, image_path):
    self.image_path = image_path
    self.rgb_image = None
    self.rgb_array = None
    self.ycc_array = None
    
  def get_ycc_subimages(self):
    if self.rgb_image is None:
        self.rgb_image = Image.open(self.image_path)
        self.rgb_array = np.asarray(self.rgb_image)

    if self.ycc_array is None:
        color_space = ColorSpace()
        red = self.rgb_array[:,:,0]
        green = self.rgb_array[:,:,1]
        blue = self.rgb_array[:,:,2]
        lum, cb, cr = color_space.to_ycc(red, green, blue)
        x,y = lum.shape
        self.ycc_array = np.zeros( (x,y,3), dtype=lum.dtype)
        self.ycc_array[:,:,0] = lum
        self.ycc_array[:,:,1] = cb
        self.ycc_array[:,:,2] = cr
    return decompose_subimages(self.ycc_array)
    
  def get_luminance_subimages(self):
      subimages = self.get_ycc_subimages()
      return np.array([i[:,:,0] for i in subimages])
      
  def get_dcts(self):
    ret_dcts = []
    for lum in self.get_luminance_subimages():
      # switched to scipy dct for performance 
      # Remember: we're subtracting 128 before DCT
      lum_dct = dct(lum - 128, type=2, norm='ortho' )
      # TODO: don't ignore the Cb, Cr channels 
      ret_dcts.append(lum_dct)
    return numpy.array(ret_dcts)


def two_dim_DCT(X, forward=True):
  """2D Discrete Cosine Transform
  X should be square 2 dimensional array
  Trying to follow:

  http://en.wikipedia.org/wiki/Discrete_cosine_transform#Multidimensional_DCTs
  http://en.wikipedia.org/wiki/JPEG#Discrete_cosine_transform

  TODO(tierney): Include precomputed results.

  precomputed = [(n1,n2, pi/N1*(n1+.5), pi/N2*(n2+0.5))
                 for n1 in range(N1) for n2 in range(N2)]

  and then your main loop is more like

  for (k1,k2),_ in numpy.ndenumerate(X):
    sub_result=0.
    for n1, n2, a1, a2 in precomputed:
      sub_result+=X[n1,n2] * cos(a1*k1) * cos(a2*k2)
    result[k1,k2] = alpha(k1) * alpha(k2) * sub_result

  you might even find that

  sub_result = sum(X[n1,n2]*cos(a1*k1)*cos(a2*k2) for
                   (n1,n2,a1,a2) in precomputed)"""

  result = zeros(X.shape)
  N1,N2 = X.shape

  def alpha(n):
    if n == 0:
      return 0.353553390593 #sqrt(1/8.)
    else:
      return .5 #sqrt(2/8.)

  for (k1,k2), _ in numpy.ndenumerate(X):
    sub_result = 0.
    if not forward:
      for n1 in range(N1):
        for n2 in range(N2):
          sub_result += alpha(n1) * alpha(n2) * X[n1,n2] * \
              cos((pi/N1)*(k1+.5)*n1) * cos((pi/N2)*(k2+.5)*n2)

              
      result[k1,k2] = sub_result
    else:
      for n1 in range(N1):
        for n2 in range(N2):
            
          sub_result += X[n1,n2] * cos((pi/N1)*(n1+.5)*k1) * \
              cos((pi/N2)*(n2+.5)*k2)
      result[k1,k2] = alpha(k1) * alpha(k2) * sub_result
  return result
