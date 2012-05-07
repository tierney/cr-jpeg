#!/usr/bin/env python

from PIL import Image
from SymbolShape import SymbolShape
from random import randint
import sys
from numpy import zeros

def print_8_by_8(pixels):
  print "Luminance"
  for row in range(8):
    for col in range(8):
      sys.stdout.write('%3d ' % pixels[col,row][0])
    sys.stdout.write('\n')

  print "Chroma Blue"
  for row in range(8):
    for col in range(8):
      sys.stdout.write('%3d ' % pixels[col,row][1])
    sys.stdout.write('\n')

  print "Chroma Red"
  for row in range(8):
    for col in range(8):
      sys.stdout.write('%3d ' % pixels[col,row][2])
    sys.stdout.write('\n')
  print


def main(argv):
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

  cb_thresholds = {
    '0': 238,
    '1': 210,
    '2': 182,
    '3': 154,
    '4': 126,
    }

  cr_thresholds = {
    '4': 126,
    '5': 98,
    '6': 70,
    '7': 42,
    '8': 14,
    }

  ss = SymbolShape([[1, 1],
                    [1, 1]])
  im = Image.new('YCbCr', (8,8))
  pixels = im.load()

  lum_orig = zeros((8,8))
  cb_orig = zeros((8,8))
  cr_orig = zeros((8,8))
  shape_width, shape_height = ss.get_shape_size()
  for base_y in range(0, 8, shape_height):
    for base_x in range(0, 8, shape_width):
      r_val_lum = thresholds[str(randint(0,8))]
      lum = r_val_lum

      cb_idx = str(randint(0,4))
      cr_idx = str(randint(4,8))
      cb = cb_thresholds[cb_idx]
      cr = cr_thresholds[cr_idx]

      for sym_i in range(ss.get_num_symbol_shapes()):
        coords = ss.get_symbol_shape_coords(sym_i+1)
        for x, y in coords:
          pixels[base_x + x, base_y + y] = (lum, cb, cr)
          lum_orig[base_y + y, base_x + x] = lum
          cb_orig[base_y + y, base_x + x] = cb
          cr_orig[base_y + y, base_x + x] = cr

  im = im.convert('YCbCr')
  pixels = im.load()
  print 'To Save:'
  print
  print_8_by_8(pixels)
  im.save('test.jpg', quality=75)

  im = Image.open('test.jpg')
  im = im.convert('YCbCr')
  pixels = im.load()
  print 'Read back image:'
  print
  print_8_by_8(pixels)

  im = im.convert("YCbCr")
  lum_extracted = zeros((8,8))
  cb_extracted = zeros((8,8))
  cr_extracted = zeros((8,8))
  for base_y in range(0, 8, shape_height):
    for base_x in range(0, 8, shape_width):
      for sym_i in range(ss.get_num_symbol_shapes()):
        val = 0
        count = 0
        coords = ss.get_symbol_shape_coords(sym_i+1)
        for x, y in coords:
          lum, cb, cr = pixels[base_x + x, base_y + y]
          lum_extracted[base_y + y, base_x + x] = lum
          cb_extracted[base_y + y, base_x + x] = cb
          cr_extracted[base_y + y, base_x + x] = cr

          val += lum
          count += 1
        print val / float(count)

  print_8_by_8(pixels)
  lum_diff = lum_extracted - lum_orig
  for base_y in range(0, 8, shape_height):
    for base_x in range(0, 8, shape_width):
      for sym_i in range(ss.get_num_symbol_shapes()):
        val = 0
        count = 0
        coords = ss.get_symbol_shape_coords(sym_i+1)
        for x, y in coords:
          val += lum_diff[base_y + y, base_x + x]
          count += 1
        print val / float(count)


  print cb_extracted - cb_orig
  print
  print cr_extracted - cr_orig
  print


if __name__ == '__main__':
  main(sys.argv)
