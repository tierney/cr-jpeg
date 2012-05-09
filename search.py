import numpy as np
from jpegcodec import JPEG

# some assumptions from looking at 
# 8x8 DCT tables of natural images
# - most coefficients are between -200 and 600
# - any coefficients in the first row are always positive 

class BitEncoder:
  def __init__(self, 
        n_bits, 
        n_control_coefs, 
        quant, 
        bits_per_bin, 
        lowest_dct_value, 
        highest_dct_value, 
        target_coef_sum):
    """
      n_bits = how many bits of information can we encode in one 8x8 matrix? 
      n_control_ceofs = how many DCT coefs get used to push the sum of coefs toward -1000?
      quant = quantization matrix
      bits_per_bin = integer vector indicating how bits can fit in each coef
      lowest_dct_value = lowest value a coef can take
      highest_dct_value = highest value a coef can take
      target_coef_sum = average of DCTs coefs in naturally occuring image patches
    """
      
    self.n_bits = n_bits
    self.n_control_coefs = n_control_coefs
    self.quant = quant
    self.ravel_quant = np.ravel(self.quant)
    self.bits_per_bin = bits_per_bin[:]
    self.num_values_per_bin = 2 ** bits_per_bin
    self.lowest_dct_value = lowest_dct_value
    self.highest_dct_value = highest_dct_value
    self.target_coef_sum = target_coef_sum


  def _gen_possible_values(self, quantization_level, n_values):
    possible_values = []
    lowest = int(self.lowest_dct_value / quantization_level) * quantization_level 
    #highest = int(self.highest_dct_value / quantization_level) * quantization_level 
    return np.arange(n_values)*quantization_level + lowest
    #while curr_value <= highest:
    #  possible_values.append(curr_value)
    #  curr_value += quantization_level
    #return possible_values 
      
  def encode(self, bits):
    """
    map a k-length bit string to a DCT matrix
    """
    assert len(bits) == self.n_bits
    
    dct = np.zeros_like(self.ravel_quant)
    running_sum = 0
    curr_input_idx = 0
    for flat_idx in np.nonzero(self.bits_per_bin)[0]:
      b = self.bits_per_bin[flat_idx]
      # map from 1D index into bits_per_bin to 
      # the 2D index into quantization and DCT matrices
      
      quantization_level = self.ravel_quant[flat_idx]
      
      # these are integers so should get rounded by division
      lowest = int(self.lowest_dct_value / quantization_level) * quantization_level 
      
      if flat_idx < 64 - self.n_control_coefs :
        
        substr = bits[curr_input_idx:curr_input_idx+b]
        # convert the boolean substring to a single decimal number
        positional_values = 2**np.arange(b)
        n = np.dot(substr, positional_values)
        v = n * quantization_level + lowest
        curr_input_idx += b 
      else:
        # we should be done with input now
        assert curr_input_idx == len(bits)
        # choose a value to make the sum of coefs closest to target sum 
        n_values = self.num_values_per_bin[flat_idx]
        possible_values = self._gen_possible_values(quantization_level, n_values)
        possible_sums = running_sum + np.array(possible_values)
        best_value_idx = np.argmin(np.abs(possible_sums - self.target_coef_sum))
        v = possible_values[best_value_idx]
      
      running_sum += v
      dct[flat_idx] = v 

    square_dct = dct.reshape(self.quant.shape)

    return square_dct
    
  def decode(self, dct):
    dct_ravel = np.ravel(dct)
    bits = np.zeros(self.n_bits, dtype='bool')

    output_idx = 0
    for flat_idx, dct_val in enumerate(dct_ravel[:-self.n_control_coefs]):
      
      quantization_level = self.ravel_quant[flat_idx]
      n_values = self.num_values_per_bin[flat_idx]
      possible_values = self._gen_possible_values(quantization_level, n_values)
      n = np.argmin(np.abs(possible_values - dct_val))
      
      b = self.bits_per_bin[flat_idx]
      
      # encode binary form of n into output vector
      for i in xrange(b):
        bits[output_idx] = n % 2
        output_idx += 1
        n /= 2 
    return bits 

import math 


# limits and target sum derived from empirical stats of some random
# image of a kid on flickr
def search(quality = 25, n_control_bins = 10, lowest = -200, highest=100, target_sum = -1000):
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
  encoder = BitEncoder(
    total_nbits, 
    n_control_bins, 
    quant, 
    bits_per_bin,
    lowest,
    highest, 
    target_sum)
  return encoder

import scipy.fftpack 
def test_random_vectors(encoder, n_iters=1000):
    n_failed = 0 
    n_clipped_below = 0
    n_clipped_above = 0
    for i in xrange(n_iters):
        vec = np.random.randn(encoder.n_bits) > 0
        dct = encoder.encode(vec)
        inverse = scipy.fftpack.idct(dct, norm='ortho')
        
        # clip 
        clipped = inverse.copy()
        bottom = inverse< -128
        clipped[bottom] = -128 
        top = inverse > 127
        clipped[top] = 127 
        n_clipped_below += (0 if np.sum(bottom) == 0 else 1)
        n_clipped_above += (0 if np.sum(top) == 0 else 1)
        dct2 = scipy.fftpack.dct(clipped, norm='ortho')
        vec2 = encoder.decode(dct2)
        n_failed += (0 if np.all(vec == vec2) else 1)
        #print "vec", vec
        #print "dct", dct
        #print "inverse", inverse
        #print "clipped", clipped
        #print "dct2", dct2
        #print "vec2", vec2
        #print "# wrong bits:", np.sum(vec != vec2)
        #print 
        #print 
        if i % (n_iters / 10) == 0:
            print "...%d of %d completed" % (i, n_iters)
    print "# clipped below: %d / %d" % (n_clipped_below, n_iters)
    print "# clipped above: %d / %d " % (n_clipped_above, n_iters)
    print "# failed: %d / %d" % (n_failed, n_iters)

