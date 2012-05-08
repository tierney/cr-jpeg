import numpy as np
from jpegcodec import JPEG

# some assumptions from looking at 
# 8x8 DCT tables of natural images
# - most coefficients are between -200 and 600
# - any coefficients in the first row are always positive 

class BitEncoder:
  def __init__(self, k, r, quant, bits_per_bin, lowest, highest ):
    self.k = k
    self.r = r 
    self.quant = quant
    self.bits_per_bin = bits_per_bin[:]
    self.lowest = lowest 
    self.highest = highest 

  def encode(self, bits):
    """
    map a k-length bit string to a DCT matrix
    """
    assert len(bits) == self.k
    dct = np.zero_like(self.quant)
    running_sum = 0
    for flat_idx in np.nonzero(self.bits_per_bin):
      b = self.bits_per_bin[flat_idx]
      substr = bits[curr_idx:curr_idx+b]
      # convert the boolean substring to a single
      # decimal number 
      n = np.dot(substr, 2**np.arange(b))
      # map from 1D index into bits_per_bin to 
      # the 2D index into quantization and DCT matrices
      idx = np.unravel_index(flat_idx, self.quant.shape)
      quantization_level = self.quant[idx]
      v = n * quantization_level + (lowest if idx[0] > 0 else 0)
      running_sum += v
      dct[idx] = v 
    return dct


import math 


# limits and target sum derived from empirical stats of some random
# image of a kid on flickr
def search(quality = 25, n_control_bins = 10, lowest = -348, highest=144, target_sum = -1000, target_std = 1300):
  """
  Inputs:
    - quality_level : int (between 1 and 100)
  Returns:
    - an encoder from k-length bit strings to 
      quantized 8x8 DCT matrices
  """
  codec = JPEG()
  quant = codec.scaled_luminance_quant_matrix(quality)
  quant_values = np.ravel(quant)
  quant_shape = quant.shape
  sorted_indices = np.argsort(quant_values)
  value_range = highest - lowest 
  bits_per_bin = np.zeros_like(quant_values, dtype='int')
  for i,idx in enumerate(sorted_indices):
    q = float(quant_values[idx])
    nbits = np.ceil(np.log2(value_range / q))
    bits_per_bin[idx] = int(nbits)
  total_nbits = np.sum(bits_per_bin[:-n_control_bins]) 
  encoder = BitEncoder(total_nbits, quant, bits_per_bin)
  return encoder
