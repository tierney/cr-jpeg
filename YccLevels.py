#!/usr/bin/env python
import heapq
import sys
from scipy.spatial.distance import pdist, euclidean, wminkowski, cosine, squareform
import numpy
from ColorSpace import ColorSpace

cs = ColorSpace()
to_ycc = cs.to_ycc
to_rgb = cs.to_rgb


def valid_rgb_channel(val):
  return val >= 0 and val <= 255

def valid_rgb(rgb):
  red, green, blue = rgb
  return valid_rgb_channel(red) and valid_rgb_channel(green) and \
         valid_rgb_channel(blue)

def enumerate_color_space():
  a = numpy.fromfunction(to_rgb, (256, 256, 256))
  return a


def _numpy_array():
  rgb = enumerate_color_space()
  red, green, blue = rgb

  lum, cb, cr = (128, 128, 218)
  print red[lum][cb][cr], green[lum][cb][cr], blue[lum][cb][cr]

def _numpy_enumerate():
  valid_limits = []
  for cb in range(256):
    for cr in range(256):
      red, green, blue = to_rgb(128, cb, cr)
      if valid_rgb_channel(red) and valid_rgb_channel(green) and \
            valid_rgb_channel(blue):
        valid_limits.append((128, cb, cr))
  print 'Pairwise Distances'
  pd = pdist(valid_limits)
  num_limits = len(valid_limits)

  print 'Organizing pdist'
  top_values = heapq.nlargest(32, pd)
  for top_value in set(top_values):
    _values = numpy.where(pd == top_value)
    print top_value, _values

    # for value in _values[0]:

    #   values / num_limits
    #   print valid_limits[value],


def parameterize(color_a, color_b, num_discretizations):
  a_y, a_cb, a_cr = color_a
  b_y, b_cb, b_cr = color_b

  lum = lambda(m) : a_y + (b_y - a_y) * m
  cb = lambda(m) : a_cb + (b_cb - a_cb) * m
  cr = lambda(m) : a_cr + (b_cr - a_cr) * m

  base = 1 / (2 * float(num_discretizations))
  lum_cb_crs = []
  # print 'Parameterization'
  for i in range(num_discretizations):
    frac = base + i * 1 / float(num_discretizations)
    _lum_cb_cr = tuple(map(round, (lum(frac), cb(frac), cr(frac))))
    # print ' ', _lum_cb_cr
    lum_cb_crs.append(_lum_cb_cr)
  return lum_cb_crs


def chrominance_at_luminance(ycc_0, ycc_1, lum):
  y0, cb0, cr0 = ycc_0
  y1, cb1, cr1 = ycc_1

  cb = cb0 + (cb1 - cb0) * ( (lum - y0) / float(y1 - y0))
  cr = cr0 + (cr1 - cr0) * ( (lum - y0) / float(y1 - y0))

  return cb, cr


def dumb_factor(n, goal):
  # Assumes factors are not equal to each other and bounded above by
  # upper_bound.
  import scipy

  comb0 = scipy.comb(n, 2)
  ret_i, ret_j = 0, 0
  for i in range(n+1):
    comb1 = scipy.comb(n-i, 2)
    for j in range(i+1, n+1):
      if goal == round(comb0 - comb1 + (j - i - 1)):
        ret_i = i
        ret_j = j
        break
  return ret_i, ret_j


def main(argv):
  pass

def get_discrete_values():
  # Colors in this dictionary are describe in YCbCr space.
  colors = {'white'  : (255, 128, 128),
            'black'  : (0, 128, 128),
            'yellow' : (226, 1, 149),
            'blue'   : (29, 255, 107),
            'red'    : (76, 85, 255),
            'cyan'   : (179, 171, 1),
            'green'  : (150, 44, 21),
            'magenta' : (105, 212, 235)}

  edges = [('cyan', 'blue'),
           ('green', 'black'),
           ('white', 'magenta'),
           ('white', 'yellow'),
           ('yellow', 'green'),
           ('yellow', 'red'),
           ('red', 'black'),
           ('magenta', 'blue'),
           ('white', 'cyan'),
           ('cyan', 'blue'),
           ('blue', 'black'),
           ('cyan', 'green'),
           ]

  edge_val_at_lum = {}

  valid_coords_for_discretization = []
  for _ycc in parameterize(colors['black'], colors['white'], 36):
    lum, _, _ = map(int, map(round, _ycc))
    print lum

    if lum < 75 or lum > 150:
      continue

    admitted_edges = []
    for edge in edges:
      c0, c1 = edge

      lum_0 = colors[c0][0]
      lum_1 = colors[c1][0]

      max_lum = max(lum_0, lum_1)
      min_lum = min(lum_0, lum_1)

      if lum >= min_lum and lum <= max_lum:
        admitted_edges.append(edge)

    print admitted_edges
    for _edge in admitted_edges:
      c0, c1 = _edge
      _cb, _cr = chrominance_at_luminance(colors[c0], colors[c1], lum)
      edge_val_at_lum[_edge] = (lum, _cb, _cr)

    points = edge_val_at_lum.values()

    for point in points:
      point = map(int, map(round, point))
      print point, valid_rgb(point)
      if valid_rgb(point):
        valid_coords_for_discretization.append(tuple(point))

  edge_vals = set(sorted(valid_coords_for_discretization))
  return list(edge_vals)


  # pd = pdist(points)
  # pdsf = squareform(pd)
  # print pdsf

  # adjacent_edges = []
  # valid_lum_edge_points = set()
  # for i, edge_a in enumerate(edges):
  #   edge_b = edges[(i + 1) % len(edges)]
  #   ycc_a = edge_val_at_lum[edge_a]
  #   ycc_b = edge_val_at_lum[edge_b]

  #   _parameterized = parameterize(ycc_a, ycc_b, 100)
  #   for _val in _parameterized:
  #     if valid_rgb(_val):
  #       valid_lum_edge_points.add(_val)
  # valid_lum_edge_points = list(valid_lum_edge_points)
  # num_edge_points = len(valid_lum_edge_points)
  # pd = pdist(list(valid_lum_edge_points))

  # top_values = heapq.nlargest(32, pd)

  # for top_value in set(top_values):
  #   print 'Distance:', top_value
  #   _values = numpy.where(pd == top_value)
  #   pairs = _values[0]
  #   for pair in pairs:
  #     print 'Goal:', pair,
  #     i, j = dumb_factor(num_edge_points, pair)
  #     print i, j, valid_lum_edge_points[i], valid_lum_edge_points[j]


if __name__ == '__main__':
  main(sys.argv)
