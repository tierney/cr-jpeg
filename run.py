#!/usr/bin/env python
from jpegcodec import JPEG
from dct import two_dim_DCT
import sys
from numpy import zeros, ones, array
import math
import random

def print_8_by_8(pixels):
  for row in range(8):
    sys.stdout.write('[ ')
    for col in range(8):
      sys.stdout.write('%9.2f ' % pixels[col,row])
      # sys.stdout.write('%4d, ' % round(pixels[col,row]))
    sys.stdout.write('],\n')

def discretize_channel(divisions):
  signal_values = []
  size = int(math.floor(256. / divisions))
  for i in range(size / 2, 256, size):
    signal_values.append(i)
  return signal_values

def random_22_block(possible_values):
  matrix = zeros((8,8))
  for i in range(0, 8, 2):
    for j in range(0, 8, 2):
      val = random.choice(possible_values)
      matrix[i,j] = val
      matrix[i+1,j] = val
      matrix[i,j+1] = val
      matrix[i+1,j+1] = val
  return matrix

def random_11_block(possible_values):
  matrix = zeros((8,8))
  for i in range(0, 8, 1):
    for j in range(0, 8, 1):
      val = random.choice(possible_values)
      matrix[i,j] = val
  return matrix

def random_sb_block(possible_values):
  matrix = zeros((8,8))
  coords = [
    [(0,0),(0,1),(1,0),(1,1),(2,0),(0,2)],
    [(6,0),(6,1),(7,0),(7,1)],
    [(0,6),(1,6),(0,7),(1,7)],
    [(6,6),(7,6),(6,7),(7,7)]]
  for coord_set in coords:
    val = random.choice(possible_values)
    for coord in coord_set:
      matrix[coord] = val
  return matrix

def main(argv):
  # discretize_channel(9)
  # discretize_channel(10)
  vals = discretize_channel(9)
  print vals

  rand_block = random_11_block(vals)
  print rand_block
  dctd = two_dim_DCT(rand_block)
  print 'DCTd'
  print_8_by_8(dctd)

  jpeg = JPEG()
  print

  # For visualizing the quantization matrices.
  #
  # for quality in range(5, 96):
  #   quantized = jpeg.scaled_luminance_quant_matrix(quality)
  #   with open('quant_%s.log' % quality, 'w') as fh:
  #     for row in range(8):
  #       for col in range(8):
  #         fh.write('%d %d %d\n' % (row, col, quantized[row,col]))

  quantized = jpeg.luminance_quantize(dctd, 75)
  print_8_by_8(quantized)
  print 'Dequanitzed'
  dequantized = jpeg.luminance_dequantize(quantized, 75)
  print_8_by_8(dequantized)
  idctd = two_dim_DCT(dequantized, False)
  print 'iDCT'
  print_8_by_8(idctd)
  print 'Diff'
  print_8_by_8( idctd - rand_block)
  print 'FS'
  dithered = jpeg.fs_dither(idctd)
  print_8_by_8(dithered)
  print 'Diff Dither'
  print_8_by_8( dithered - rand_block)

  # matrix = zeros((8,8))
  # print matrix


  # sqm = jpeg.scaled_luminance_quant_matrix(76)
  # print_8_by_8(sqm)
  # print
  # m = 1400. / sqm
  # print_8_by_8(m)
  # print sum(sum(m))

  # to_idct = sqm

  # to_idct = 255 * ones((8,8)) # dequantized matrix.
  # idct_sqm = two_dim_DCT(to_idct, False)
  # print 'IDCT'
  # print_8_by_8(idct_sqm)
  # print_8_by_8(jpeg.luminance_quantize(idct_sqm, 76))
  # print

  # print_8_by_8(two_dim_DCT(idct_sqm))

  # thresholds = { # Well, nine (we add one for "black").
  #   '0': 238,
  #   '1': 210,
  #   '2': 182,
  #   '3': 154,
  #   '4': 126,
  #   '5': 98,
  #   '6': 70,
  #   '7': 42,
  #   '8': 14,
  #   }

  # mat = array([
  #     [2 ,  -1,  1,  0,   0,   0,   0,   0],
  #     [-1 ,  0,  0,  0,   0,   0,   0,   0],
  #     [1    ,  0,  0,   0,   0,   0,   0,   0],
  #     [0    ,  0,   0,   0,   0,   0,   0,   0],
  #     [ 0   ,   0,   0,   0,   0,   0,   0,   0],
  #     [ 0   ,   0,   0,   0,   0,   0,   0,   0],
  #     [ 0   ,   0,   0,   0,   0,   0,   0,   0],
  #     [ 0   ,   0,   0,   0,   0,   0,   0,   0]])
  # # Dequantize. Then IDCT
  # print
  # print two_dim_DCT(mat, False) + 128


if __name__ == '__main__':
  main(sys.argv)
