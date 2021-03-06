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
  increment = 48
  with open('rgb_ycc.txt', 'w') as fh:
    for red in range(0, 256, increment):
      for green in range(0, 256, increment):
        for blue in range(0, 256, increment):
          lum, cb, cr = cs.to_ycc(red, green, blue)
          fh.write('%d,%d,%d,%.2f,%.2f,%.2f\n' % \
                   (red, green, blue, lum, cb, cr))
    fh.flush()


def hex_to_rgb(value):
  value = value.lstrip('#')
  lv = len(value)
  return tuple(int(value[i:i + lv / 3], 16) for i in range(0, lv, lv / 3))


def rgb_to_hex(rgb):
  return '#%02x%02x%02x' % rgb


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


def visualize_ycc():
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
      ax.scatter(cb, cr, lum, c=color, s=40)

  ax.set_xlabel('Chroma Blue')
  ax.set_ylabel('Chroma Red')
  ax.set_zlabel('Luminance')

  plt.show()
  im = Image.new('RGB', (8, 8))
  pixels = im.load()

  array = numpy.zeros((8, 8, 3))


def parameterize(color_a, color_b, num_discretizations, base=None):
  a_y, a_cb, a_cr = color_a
  b_y, b_cb, b_cr = color_b

  lum = lambda(m) : a_y + (b_y - a_y) * m
  cb = lambda(m) : a_cb + (b_cb - a_cb) * m
  cr = lambda(m) : a_cr + (b_cr - a_cr) * m

  if not base:
    base = 1 / (2 * float(num_discretizations))

  lum_cb_crs = []
  print 'Parameterization'
  for i in range(num_discretizations):
    frac = base + i * 1 / float(num_discretizations)
    _lum_cb_cr = tuple(map(round, (lum(frac), cb(frac), cr(frac))))
    print ' ', _lum_cb_cr
    lum_cb_crs.append(_lum_cb_cr)
  return lum_cb_crs


def main(argv):
  # visualize_ycc()
  # return

  # Parse arguments.
  parser = argparse.ArgumentParser(
    prog='SixBits', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  args = parser.parse_args()
  _ = args

  # Colors in this dictionary are describe in YCbCr space.
  color = {'white'  : (255, 128, 128),
           'black'  : (0, 128, 128),
           'yellow' : (226, 1, 149),
           'blue'   : (29, 255, 107),
           'red'    : (76, 85, 255),
           'cyan'   : (179, 171, 1),
           'green'  : (150, 44, 21),
           'magenta' : (105, 212, 235)}

  all_values = []

  num_discretizations = 9
  side_discrets = 4

  import YccLevels
  all_values = YccLevels.get_discrete_values()
  all_values = list(set(all_values))
  # to_remove = [
  #   (100,  72,  57),
  #   (132,  54, 215),
  #   (147,  45, 205),
  #   (100, 215,  57),
  #   ( 85, 224,  67),
  #   (132, 197, 216),
  #   (147, 188, 205)
  #   ]
  # for _ in to_remove:
  #   all_values.remove(_)

  # all_values.remove((128, 128, 128)) # Remove most ambiguious chunks.

  print 'Values'
  for _val in sorted(all_values):
    print ' ', _val

  num_values = len(all_values)
  import math
  print 'Number of distinct values: %d (%.2f bits).' % \
    (num_values, math.log(num_values, 2))
  from scipy.spatial.distance import pdist, euclidean, wminkowski, cosine
  distances = pdist(all_values)
  count = 0
  idx_val = {}
  for i in range(num_values):
    for j in range(i + 1, num_values):
      idx_val[count] = (i, j)
      count += 1

  # for idx in idx_val:
  #   first, second = idx_val[idx]
  #   if distances[idx] < 28:
  #     print idx, all_values[first], all_values[second], distances[idx]

  possible_values = all_values
  IMAGE_WIDTH = 8
  IMAGE_HEIGHT = 8
  im = Image.new('YCbCr', (IMAGE_WIDTH, IMAGE_HEIGHT))
  pixels = im.load()
  original_vals = []
  for row in range(0, IMAGE_HEIGHT, 2):
    for col in range(0, IMAGE_WIDTH, 2):
      val = tuple(map(int, random.choice(possible_values)))
      original_vals.append(val)
      print 'Written:', val
      pixels[row, col] = val
      pixels[row + 1, col] = val
      pixels[row, col + 1] = val
      pixels[row + 1, col + 1] = val
  original_vals.reverse()

  QUALITY = 75
  im.save('test.jpg', quality = QUALITY)

  print 'ALL VALUES', all_values

  opened_im = Image.open('test.jpg')
  pixels = opened_im.load()
  for row in range(0, IMAGE_HEIGHT, 2):
    for col in range(0, IMAGE_WIDTH, 2):
      vals = {}
      for idx in range(3):
        val = 0
        val += pixels[row, col][idx]
        val += pixels[row + 1, col][idx]
        val += pixels[row, col + 1][idx]
        val += pixels[row + 1, col + 1][idx]
        val /= 4.0
        vals[idx] = val

      red = vals[0]
      green = vals[1]
      blue = vals[2]
      extracted = ColorSpace.to_ycc(red, green, blue)
      print ' Extracted', extracted
      _best_val = 1000
      _best_match = ()

      for vect in all_values:
        vect = map(int, map(round, vect))
        dist = wminkowski(extracted, vect, 2, [5, 1, 1])
        print extracted, vect, dist
        if dist < _best_val:
          _best_val = dist
          _best_match = vect
      print ' Best Val', _best_val, _best_match
      _orig = list(original_vals.pop())
      _best_match = map(int, _best_match) # map(int, _min_vect)

      if _orig != _best_match:
        _mismatch_print = 'Mismatch at (%3d, %3d):\n' % (row, col)
        orig_extracted_dist = wminkowski(_orig, extracted, 2, [5, 1, 1])
        _mismatch_print += '  original  = (%3d, %3d, %3d) %6.2f\n' % tuple(_orig + [orig_extracted_dist])
        best_match_dist = wminkowski(_orig, _best_match, 2, [5, 1, 1])
        _mismatch_print += '  closest   = (%3d, %3d, %3d) %6.2f\n' % tuple(_best_match + [best_match_dist])
        _mismatch_print += '  extracted = (%3d, %3d, %3d)\n' % tuple(map(int, map(round, extracted)))

        print _mismatch_print
        # print >>sys.stderr, _mismatch_print


if __name__ == '__main__':
  main(sys.argv)
