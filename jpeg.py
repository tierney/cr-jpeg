#!/usr/bin/env python

std_luminance_quant_tbl = [
  16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99
  ]

chrominance = [
  17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
  ]

import numpy
#shortcuts
from numpy import pi
from math import cos,sqrt

def two_dim_DCT(X):
  """2D Discrete Cosine Transform
  X should be square 2 dimensional array
  Trying to follow:

  http://en.wikipedia.org/wiki/Discrete_cosine_transform#Multidimensional_DCTs

  http://en.wikipedia.org/wiki/JPEG#Discrete_cosine_transform

precomputed = [(n1,n2, pi/N1*(n1+.5), pi/N2*(n2+0.5))
for n1 in range(N1) for n2 in range(N2)]

and then your main loop is more like

for (k1,k2),_ in numpy.ndenumerate(X):
sub_result=0.
for n1, n2, a1, a2 in precomputed:
sub_result+=X[n1,n2] * cos(a1*k1) * cos(a2*k2)
result[k1,k2]=alpha(k1)*alpha(k2)*sub_result

you might even find that

sub_result = sum(X[n1,n2]*cos(a1*k1)*cos(a2*k2) for (n1,n2,a1,a2) in precomputed)
"""
  result = numpy.zeros(X.shape)
  N1,N2 = X.shape

  def alpha(n):
    """Normalizing function, not sure if necessary"""
    if n == 0:
      return 0.353553390593 #sqrt(1/8.)
    else:
      return .5 #sqrt(2/8.)

  for (k1,k2), _ in numpy.ndenumerate(X):
    sub_result = 0.
    for n1 in range(N1):
      for n2 in range(N2):
        sub_result += X[n1,n2] * cos((pi/8.)*(n1+.5)*k1)*cos((pi/8.)*(n2+.5)*k2)
    result[k1,k2] = alpha(k1) * alpha(k2) * sub_result
  return result

def jpeg_quality_scaling(quality):
  # From jcparams.c:123
  if quality <= 0:
    quality = 1
  if quality > 100:
    quality = 100

  if quality < 50:
    quality = 5000 / quality # int in the C code.
  else:
    quality = 200 - quality*2

  return quality

# for i in range(0, 101):
#   print i, jpeg_quality_scaling(i)


# import random
# mat = numpy.zeros((8,8))
# for row in range(8):
#   for col in range(8):
#     mat[row,col] = random.randint(0, 255)

mat_wiki = [[52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94]]

mat = numpy.array(mat_wiki)
# Shift for [-128, 127]
mat -= 128

for row in range(8):
  for col in range(8):
    print '%8.2f' % mat[row, col],
  print ''
print

dct_mat = two_dim_DCT(mat)
for row in range(8):
  for col in range(8):
    print '%8.2f' % dct_mat[row, col],
  print ''
print

quantized_mat = numpy.zeros((8,8))
for i, quant in enumerate(std_luminance_quant_tbl):
  row = i / 8
  col = i % 8
  quantized_mat[row, col] = round(dct_mat[row, col] / float(quant))

for row in range(8):
  for col in range(8):
    print '%3d' % quantized_mat[row, col],
  print ''
