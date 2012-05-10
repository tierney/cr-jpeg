#!/usr/bin/env python
import heapq
import sys
from scipy.spatial.distance import pdist, euclidean, wminkowski, cosine
import numpy
from ColorSpace import ColorSpace

cs = ColorSpace()
to_ycc = cs.to_ycc
to_rgb = cs.to_rgb

def valid_rgb_channel(val):
  return val >= 0 and val <= 255

def enumerate_color_space():
  a = numpy.fromfunction(to_rgb, (256, 256, 256))
  return a

def _numpy_array():
  rgb = enumerate_color_space()
  red, green, blue = rgb

  lum, cb, cr = (128, 128, 218)
  print red[lum][cb][cr], green[lum][cb][cr], blue[lum][cb][cr]


def create_top_k_list(values):
  # values is a list where the index is what we want to keep track of.
  pass

def main(argv):
  valid_limits = []
  for cb in range(256):
    for cr in range(256):
      red, green, blue = to_rgb(128, cb, cr)
      if valid_rgb_channel(red) and valid_rgb_channel(green) and valid_rgb_channel(blue):
        valid_limits.append((128, cb, cr))
  print 'Pairwise Distances'
  pd = pdist(valid_limits)

  print 'Organizing pdist'
  top_values = heapq.nlargest(32, pd)
  for top_value in top_values:
    print top_value, numpy.where(pd == top_value)

if __name__ == '__main__':
  main(sys.argv)
