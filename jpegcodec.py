#!/usr/bin/env python

import numpy

class JPEG(object):

  _std_luminance_quant_tbl = numpy.array([
      [16,  11,  10,  16,  24,  40,  51,  61],
      [12,  12,  14,  19,  26,  58,  60,  55],
      [14,  13,  16,  24,  40,  57,  69,  56],
      [14,  17,  22,  29,  51,  87,  80,  62],
      [18,  22,  37,  56,  68, 109, 103,  77],
      [24,  35,  55,  64,  81, 104, 113,  92],
      [49,  64,  78,  87, 103, 121, 120, 101],
      [72,  92,  95,  98, 112, 100, 103,  99]])

  _std_chrominance_quant_tbl = numpy.array([
      [17,  18,  24,  47,  99,  99,  99,  99],
      [18,  21,  26,  66,  99,  99,  99,  99],
      [24,  26,  56,  99,  99,  99,  99,  99],
      [47,  66,  99,  99,  99,  99,  99,  99],
      [99,  99,  99,  99,  99,  99,  99,  99],
      [99,  99,  99,  99,  99,  99,  99,  99],
      [99,  99,  99,  99,  99,  99,  99,  99],
      [99,  99,  99,  99,  99,  99,  99,  99]])

  @staticmethod
  def scale_factor(quality):
    # From jcparams.c:123
    if quality <= 0:
      quality = 1
    if quality > 100:
      quality = 100

    if quality < 50:
      quality = 5000. / quality # int in the C code.
    else:
      quality = 200. - quality*2

    return quality

  def _scaled_quant_matrix(self, quality, std_quant_tbl):
    _quant_table = numpy.zeros((8,8))
    _scale_factor = self.scale_factor(quality)
    for row in range(8):
      for col in range(8):
        temp = (float(_scale_factor) * std_quant_tbl[row,col] \
                  + 50.) / 100.
        if temp <= 0:
          temp = 1.
        if temp > 32767:
          temp = 32767.
        if temp > 255:
          temp = 255. # TODO(tierney): If forcing baseline.
        _quant_table[row,col] = int(temp)
    with open('quant.log','w') as fh:
      for row in range(8):
        for col in range(8):
          fh.write('%d %d %d\n' % (row, col, _quant_table[row,col]))
    return _quant_table

  def scaled_luminance_quant_matrix(self, quality):
    return self._scaled_quant_matrix(quality, self._std_luminance_quant_tbl)

  def scaled_chrominance_quant_matrix(self, quality):
    return self._scaled_quant_matrix(quality, self._std_chrominance_quant_tbl)

  def luminance_quantize(self, dct_mat, quality):
    quant = self.scaled_luminance_quant_matrix(quality)
    ret = numpy.zeros((8,8))
    for row in range(8):
      for col in range(8):
        quant_val = int(round(dct_mat[row,col] / float(quant[row,col])))
        ret[row,col] = quant_val
    return ret

  def luminance_dequantize(self, dezz, quality):
    quant = self.scaled_luminance_quant_matrix(quality)
    print quant
    ret = numpy.zeros((8,8))
    for row in range(8):
      for col in range(8):
        ret[row, col] = dezz[row,col] * quant[row,col]
    return ret

  def chrominance_quantize(self, dct_mat, quality):
    quant = self.scaled_chrominance_quant_matrix(quality)
    ret = numpy.zeros((8,8))
    for row in range(8):
      for col in range(8):
        quant_val = int(round(dct_mat[row,col] / float(quant[row,col])))
        ret[row,col] = quant_val
    return ret
