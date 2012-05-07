#!/usr/bin/env python
import numpy
from numpy import pi,zeros
from math import cos,sqrt
from PIL import Image
from ColorSpace import ColorSpace

class DCT(object):
  # Example use:
  #   from dct import DCT
  #   dct = DCT('test.jpg',75)
  #   ret = dct.get_dcts()
  def __init__(self, image_path, quality):
    self.image_path = image_path
    self.quality = quality

  def _decompose(self):
    # Imperfectly grabs all of the 8x8 pixel blocks (will ignore edge blocks
    # that are smaller than 8x8).
    im = Image.open(self.image_path)
    width, height = im.size

    width_intevals = width / 8 # Leverage automatic floor-ing of divided ints.
    height_intervals = height / 8

    sub_images = []
    for height_interval in range(width_intevals):
      row_sub_images = []
      for width_interval in range(height_intervals):
        box = (width_interval * 8, height_interval * 8,
               (width_interval + 1) * 8, (height_interval + 1) * 8)
        sub_image = im.crop(box)
        pixels = numpy.array(sub_image)
        row_sub_images.append(pixels)
      sub_images.append(row_sub_images)
    return sub_images

  def get_dcts(self):
    color_space = ColorSpace()
    sub_images = self._decompose()

    ret_dcts = []
    for sub_image_row in sub_images:
      ret_dcts_row = []
      for pixels in sub_image_row:
        _lum_pixels = numpy.zeros((8,8))
        _cb_pixels = numpy.zeros((8,8))
        _cr_pixels = numpy.zeros((8,8))
        for row in range(8):
          for col in range(8):
            red, green, blue = pixels[row,col]
            lum, cr, cb = color_space.to_ycc(red, green, blue)
            _lum_pixels[row,col] = lum
            _cb_pixels[row,col] = cb
            _cr_pixels[row,col] = cr

        lum_dct = two_dim_DCT(_lum_pixels)
        # TODO(tierney): Technically, need to subsample Cb and Cr before
        # DCT. These Cb and Cr values should be treated as unrealistic until
        # subsampling before the DCT step.
        cb_dct = two_dim_DCT(_cb_pixels)
        cr_dct = two_dim_DCT(_cr_pixels)

        ret_dcts_row.append(lum_dct)
      ret_dcts.append(ret_dcts_row)
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
    """Normalizing function, not sure if necessary"""
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
