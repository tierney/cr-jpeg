#!/usr/bin/env python

from ColorSpace import ColorSpace
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random
import sys

def generate_ycc_from_rgb():
  cs = ColorSpace()
  increment = 24
  with open('rgb_ycc.txt', 'w') as fh:
    for red in range(0, 256, increment):
      for green in range(0, 256, increment):
        for blue in range(0, 256, increment):
          lum, cb, cr = cs.to_ycc(red, green, blue)
          fh.write('%d,%d,%d,%.2f,%.2f,%.2f\n' % (red, green, blue, lum, cb, cr))
    fh.flush()


def hex_to_rgb(value):
  value = value.lstrip('#')
  lv = len(value)
  return tuple(int(value[i:i + lv / 3], 16) for i in range(0, lv, lv / 3))


def rgb_to_hex(rgb):
  return '#%02x%02x%02x' % rgb


def parameterize(color_a, color_b, num_discretizations):
  a_y, a_cb, a_cr = color_a
  b_y, b_cb, b_cr = color_b

  lum = lambda(m) : a_y + (b_y - a_y) * m
  cb = lambda(m) : a_cb + (b_cb - a_cb) * m
  cr = lambda(m) : a_cr + (b_cr - a_cr) * m

  base = 1 / (2 * float(num_discretizations))
  lum_cb_crs = []
  for i in range(num_discretizations):
    frac = base + i * 1 / float(num_discretizations)
    _lum_cb_cr = tuple(map(round, (lum(frac), cb(frac), cr(frac))))
    lum_cb_crs.append(_lum_cb_cr)
  return lum_cb_crs


def random_22_block(possible_values):
  matrix = numpy.zeros((8, 8, 3))
  for i in range(0, 8, 2):
    for j in range(0, 8, 2):
      val = random.choice(possible_values)
      print val
      matrix[i, j] = val
      matrix[i + 1, j] = val
      matrix[i, j + 1] = val
      matrix[i + 1, j + 1] = val
  return matrix

def main(argv):
  # Parse arguments.
  parser = argparse.ArgumentParser(
    prog='SixBits', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  args = parser.parse_args()
  _ = args

  color = {'white'  : (255, 128, 128),
           'black'  : (0, 128, 128),
           'yellow' : (226, 1, 149),
           'blue'   : (29, 255, 107),
           'red'    : (76, 85, 255),
           'cyan'   : (179, 171, 1),
           'green'  : (150, 44, 21),
           'purple' : (105, 212, 235)}

  all_values = []
  all_values += parameterize(color['white'], color['black'], 9)
  all_values += parameterize(color['yellow'], color['blue'], 5)
  all_values += parameterize(color['red'], color['cyan'], 5)
  all_values += parameterize(color['green'], color['purple'], 5)
  num_values = len(all_values)
  from scipy.spatial.distance import pdist
  distances = pdist(all_values)
  count = 0
  idx_val = {}
  for i in range(num_values):
    for j in range(i + 1, num_values):
      idx_val[count] = (i, j)
      count += 1
  for idx in idx_val:
    first, second = idx_val[idx]
    if distances[idx] < 28:
      print idx, all_values[first], all_values[second], distances[idx]
  return

  im = Image.new('YCbCr', (8, 8))
  pixels = im.load()
  for row in range(0, 8, 2):
    for col in range(0, 8, 2):
      val = tuple(map(int, random.choice(possible_values)))
      print val
      pixels[row, col] = val
      pixels[row + 1, col] = val
      pixels[row, col + 1] = val
      pixels[row + 1, col + 1] = val

  im.save('test.jpg', quality=95)

  parameterize(color['yellow'], color['blue'], 9)
  parameterize(color['red'], color['cyan'], 9)
  parameterize(color['green'], color['purple'], 9)


  return
  #scatter3d_demo()
  generate_ycc_from_rgb()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  with open('rgb_ycc.txt') as fh:
    count = 0
    while True:
      line = fh.readline()
      count += 1
      if not line:
        break
      if count % 1000 == 0:
        print count
      red, green, blue, lum, cb, cr = line.split(',')
      [lum, cb, cr] = map(float, [lum, cb, cr])
      color = rgb_to_hex(tuple(map(int, (red, green, blue))))
      mapped = [[x] for x in [lum, cb, cr]]
      lum, cb, cr = map(numpy.array, mapped)
      ax.scatter(cb, cr, lum, c=color)

  ax.set_xlabel('Chroma Blue')
  ax.set_ylabel('Chroma Red')
  ax.set_zlabel('Luminance')

  plt.show()
  im = Image.new('RGB', (8, 8))
  pixels = im.load()

  array = numpy.zeros((8, 8, 3))


if __name__ == '__main__':
  main(sys.argv)
