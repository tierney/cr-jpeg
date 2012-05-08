import numpy as np
from jpegcodec import JPEG

# some assumptions from looking at 
# 8x8 DCT tables of natural images
# - most coefficients are between -200 and 600
# - any coefficients in the first row are always positive 
LOWEST = -200
HIGHEST = 200 

class BitEncoder:
  def __init__(self, k, quant, bits_per_bin):
    self.k = k
    self.quant = quant
    self.bits_per_bin = bits_per_bin[:]

  def encode(self, bits):
    """
    map a k-length bit string to a DCT matrix
    """
    assert len(bits) == self.k
    dct = np.zero_like(self.quant)
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
      v = n * quantization_level + (LOWEST if idx[0] > 0 else 0)
      dct[idx] = v 
    return dct



def search(quality = 20):
  """
  Inputs:
    - quality_level : int (between 1 and 100)
  Returns:
    - an encoder from k-length bit strings to 
      quantized 8x8 DCT matrices
  """
  codec = JPEG()
  quant = codec.scaled_luminance_quant_matrix(quality)
  quant_values = quant[:]
  quant_shape = quant.shape
  sorted_indices = np.argsort(quant_values)
  bits_per_bin = np.zeros_like(quant_values, dtype='int')
  for i,idx in enumerate(sorted_indices):
    q = quant_values[idx]
    nbits = HIGHEST / q     
    bits_per_bin[idx] = nbits
  total_nbits = np.sum(bits_per_bin) 
  encoder = BitEncoder(total_nbits, quant, bits_per_bin)
  return encoder
