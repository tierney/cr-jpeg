#!/usr/bin/env python
import numpy
from numpy import pi,zeros
from math import cos,sqrt

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
