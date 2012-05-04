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

std_chrominance_quant_tbl = [
  17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
  ]

_QUANT_TABLE = std_luminance_quant_tbl

_QUALITY = 72
_PIL_QUALITY = _QUALITY

import numpy
#shortcuts
from numpy import pi
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


def jpeg_scale_factor(quality):
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

import random
thresholds = { # Well, nine (we add one for "black").
  '0': 238,
  '1': 210,
  '2': 182,
  '3': 154,
  '4': 126,
  '5': 98,
  '6': 70,
  '7': 42,
  '8': 14,
  }
_inv_thresholds = dict((v,k) for k, v in thresholds.iteritems())

from main import ColorSpace

b64_alphabet = \
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

message = ''
message_base8 = ''
indices = []
values = []
for i in range(32):
  idx = random.randint(0,63)
  message += b64_alphabet[idx]
  indices.append(idx)
  octal_val = '%02s' % oct(idx)[1:]
  message_base8 += octal_val.replace(' ','0')

print message
print message_base8, len(message_base8)
print indices

mat = numpy.zeros((8,8))
img_buffer = []

print 'Original RGB'
for row in range(0, 8, 1):
  for col in range(0, 8, 1):
    value = thresholds[message_base8[row*8 + col]]
    red = green = blue = value
    print '%5.1f' % value,
    y, cb, cr = ColorSpace.to_ycc(red, green, blue)
    mat[row,col] = float(y)
    img_buffer.append(int(y))
    # mat[row+1,col] = float(y)
    # mat[row,col+1] = float(y)
    # mat[row+1,col+1] = float(y)
  print
print

from PIL import Image
print str(img_buffer)
im = Image.new('RGB', (16,16))
pixels = im.load()
# for i in range(0, 16, 2):
#   for j in range(0, 16, 2):
#     rgb = (img_buffer[(i/2)*8 + (j/2)],
#            img_buffer[(i/2)*8 + (j/2)],
#            img_buffer[(i/2)*8 + (j/2)])
#     pixels[i,j] = rgb
#     pixels[i+1,j] = rgb
#     pixels[i,j+1] = rgb
#     pixels[i+1,j+1] = rgb

for i in range(0, 8, 1):
  for j in range(0, 8, 1):
    rgb = (img_buffer[(i)*8 + (j)],
           img_buffer[(i)*8 + (j)],
           img_buffer[(i)*8 + (j)])
    pixels[i,j] = rgb

im.save('test.jpg', quality=_PIL_QUALITY)

mat_wiki = [[52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94]]

# orig_mat = numpy.array(mat_wiki)
orig_mat = mat
print "Color space converted original."
for row in range(8):
  for col in range(8):
    print '%5.1f' % orig_mat[row, col],
  print ''
print


# Shift for [-128, 127]
mat = orig_mat - 128

for row in range(8):
  for col in range(8):
    print '%4d' % mat[row, col],
  print ''
print

dct_mat = two_dim_DCT(mat)
for row in range(8):
  for col in range(8):
    print '%8.2f' % dct_mat[row, col],
  print ''
print

scale_factor = jpeg_scale_factor(_QUALITY)

print scale_factor
quantized_mat = numpy.zeros((8,8))
for i, quant in enumerate(_QUANT_TABLE):
  row = i / 8
  col = i % 8

  # temp = float(quant)
  temp = ((scale_factor * float(quant) + 50.) / 100.)
  if temp <= 0:
    print 'forcing positive'
    temp = 1.

  if temp > 32767:
    print 'forcing trim'
    temp = 32767.
  if temp > 255:
    print 'forcing baseline'
    temp = 255. # TODO(tierney): If forcing baseline.

  quantized_mat[row, col] = \
      round(dct_mat[row, col] / float(temp))

print 'Quantized Luminance'
for row in range(8):
  for col in range(8):
    print '%5.1f' % quantized_mat[row, col],
  print ''
print


dequantized_mat = numpy.zeros((8,8))
for i, quant in enumerate(_QUANT_TABLE):
  row = i / 8
  col = i % 8

  # temp = float(quant)
  temp = (quant * scale_factor + 50.) / 100.
  dequantized_mat[row, col] = \
      quantized_mat[row, col] * temp

print 'Dequantized'
for row in range(8):
  for col in range(8):
    print '%4d' % dequantized_mat[row, col],
  print ''
print


idct_mat = two_dim_DCT(dequantized_mat, False)
print 'IDCT of Dequantized'
for row in range(8):
  for col in range(8):
    print '%4d' % round(idct_mat[row, col]),
  print ''
print

recovered_mat = numpy.zeros((8,8))
idct_mat += 128
from util import bsearch

_inv_keys = sorted(_inv_thresholds.keys())
ret = []
print 'Recovered Matrix'
for row in range(8):
  for col in range(8):
    recovered_val = round(idct_mat[row, col])
    ret.append(int(_inv_thresholds[
          _inv_keys[bsearch(_inv_keys, recovered_val)]]))
    print '%4d' % recovered_val,
    recovered_mat[row, col] = recovered_val
  print ''
print


# Recover the real image that was written.
recovered_image = Image.open('test.jpg')
recovered_pixels = recovered_image.load()
recovered_img_ret = []
print
print 'Recovered Image Real'
# for i in range(0, 16, 2):
#   for j in range(0, 16, 2):
#     avg = 0.
#     for x, y in [(0,0), (0,1), (1,0), (1,1)]:
#       avg += recovered_pixels[i + x,j + y][0]
#     recovered_val = avg / 4.
#     print '%6.1f' % recovered_val,
#     recovered_img_ret.append(int(_inv_thresholds[
#           _inv_keys[bsearch(_inv_keys, recovered_val)]]))
#   print
for i in range(0, 8, 1):
  for j in range(0, 8, 1):
    avg = 0.
    for x, y in [(0,0)]:
      avg += recovered_pixels[i + x,j + y][0]
    recovered_val = avg / 1.
    print '%6.1f' % recovered_val,
    recovered_img_ret.append(int(_inv_thresholds[
          _inv_keys[bsearch(_inv_keys, recovered_val)]]))
  print

recovered_img_message = ''.join([str(i) for i in recovered_img_ret])
print 'Recovered img'
print recovered_img_message
errors = [abs(int(message_base8[i]) - int(recovered_img_message[i]))
          for i in range(64)]
print ''.join([str(i) for i in errors])
num_errors = sum([errors[i] for i in range(64)])
print num_errors
print

print 'Recovered libjpeg'
recovered_message = ''.join([str(i) for i in ret])
print message_base8
print recovered_message
errors = [abs(int(message_base8[i]) - int(recovered_message[i]))
          for i in range(64)]
print ''.join([str(i) for i in errors])
num_errors = sum([errors[i] for i in range(64)])
print num_errors

# print recovered_mat

error = numpy.zeros((8,8))
avg_errors = 0.
for row in range(8):
  for col in range(8):
    avg_errors += abs(orig_mat[row, col] - recovered_mat[row, col])
    error[row, col] = orig_mat[row, col] - recovered_mat[row, col]

print avg_errors / 64
for row in range(8):
  for col in range(8):
    print '%4d' % error[row, col],
  print
