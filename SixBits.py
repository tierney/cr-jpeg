#!/usr/bin/env python

import sys
import argparse
from PIL import Image
import numpy
from ColorSpace import ColorSpace
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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

  for i in range(num_discretizations):
    frac = i * 1 / float(num_discretizations)
    print '%6.2f %6.2f %6.2f' % (lum(frac), cb(frac), cr(frac))


def main(argv):
  # Parse arguments.
  parser = argparse.ArgumentParser(
    prog='SixBits', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  args = parser.parse_args()
  _ = args

  color = {'white': (255, 128, 128),
           'black': (0, 128, 128),
           'yellow': (226, 1, 149),
           'blue': (29, 255, 107),
           'red': (76, 85, 255),
           'cyan': (179, 171, 1),
           'green': (150, 44, 21),
           'purple': (105, 212, 235)}

  parameterize(color['white'], color['black'], 9)
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
