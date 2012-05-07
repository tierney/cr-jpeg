import numpy as np
from jpegcodec import JPEG

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
      dct[idx] = n * quantization_level
    return dct



def search(accept, quality = 20):
  """
  Inputs:
    - accept : (bit string -> DCT Matrix) -> bool
      Repeatedly test whether a given DCT encoder survives 
      the conversions to/from RGB space.

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
  k = 0
  for i,idx in enumerate(sorted_indices):
    q = quant_values[idx]
    accepted = True
    nbits = 0
    while accepted:
      bits_per_bin[idx] = nbits
      encoder = BitEncoder(k+nbits, quant, bits_per_bin)
      accepted = accept(encoder.encode)
      # catch any runaway loops
      nbits += 1
      assert (nbits < 100)
    # last iteration failed, so use one less bit in this bin
    bits_per_bin[idx] = nbits - 1 
    k += (nbits - 1)
  return encoder
