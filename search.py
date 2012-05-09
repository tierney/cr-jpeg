import numpy as np
from jpegcodec import JPEG

# some assumptions from looking at 
# 8x8 DCT tables of natural images
# - most coefficients are between -200 and 600
# - any coefficients in the first row are always positive 
import scipy.fftpack 
class BitEncoder:
  def __init__(self, 
        n_bits,  
        quant, 
        bits_per_bin, 
        lowest_dct_value, 
        highest_dct_value):
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
    self.quant = quant
    self.ravel_quant = np.ravel(self.quant)
    self.bits_per_bin = bits_per_bin[:]
    self.num_values_per_bin = 2 ** bits_per_bin
    self.lowest_dct_value = lowest_dct_value
    self.highest_dct_value = highest_dct_value
    
  def _gen_possible_values(self, quantization_level, n_values, lowest = None):
    possible_values = []
    if lowest is None:
        lowest = int(self.lowest_dct_value / quantization_level) * quantization_level 
    else: 
        lowest = int(lowest / quantization_level) * quantization_level
    return np.arange(n_values)*quantization_level + lowest
    

  """      #n_values = self.num_values_per_bin[flat_idx]
            possible_values = self._gen_possible_values(quantization_level, 20, -300)
            # choose a value which makes the inverse as close to target clipping range as possible 
            
            best_clipping_score = np.inf
            best_value = None
            
            for candidate in possible_values:
              dct[flat_idx] = candidate
              # perform inverse DCT
              if score < best_clipping_score:
                 best_value = candidate
                 best_clipping_score = score
  """
  def _suboptimality(self, ravel_dct):
    inverse = scipy.fftpack.dct(ravel_dct.reshape((8,8)), type=3, norm='ortho')
    inverse = np.ravel(inverse)
    too_low = inverse < -128
    too_high = inverse > 127
    return np.sum(-inverse[too_low] - 128) + np.sum(inverse[too_high] - 127)
      
  def encode(self, bits):
    """
    map a k-length bit string to a DCT matrix
    """
    assert len(bits) == self.n_bits
    
    dct = np.zeros_like(self.ravel_quant)
    
    curr_input_idx = 0
    for flat_idx in np.nonzero(self.bits_per_bin)[0]:
      b = self.bits_per_bin[flat_idx]
      # map from 1D index into bits_per_bin to 
      # the 2D index into quantization and DCT matrices
      if b > 0:
          quantization_level = self.ravel_quant[flat_idx]
          
          # these are integers so should get rounded by division
          lowest = int(self.lowest_dct_value / quantization_level) * quantization_level 
          
          substr = bits[curr_input_idx:curr_input_idx+b]
          # convert the boolean substring to a single decimal number
          positional_values = 2**np.arange(b)
          if len(substr) < b:
              z = np.zeros(b, dtype='bool')
              z[:len(substr)] = substr
              substr = z
          n = np.dot(substr, positional_values)
         
          dct[flat_idx] = n * quantization_level + lowest
          curr_input_idx += b 
    square_dct = dct.reshape(self.quant.shape)
    
    return square_dct
    
  def decode(self, dct):
    dct_ravel = np.ravel(dct)
    bits = np.zeros(self.n_bits, dtype='bool')

    output_idx = 0
    for flat_idx, dct_val in enumerate(dct_ravel):
      
      quantization_level = self.ravel_quant[flat_idx]
      n_values = self.num_values_per_bin[flat_idx]
      if n_values > 0:
        possible_values = self._gen_possible_values(quantization_level, n_values)
        n = np.argmin(np.abs(possible_values - dct_val))
        b = self.bits_per_bin[flat_idx]
      
        # encode binary form of n into output vector
        for i in xrange(b):
          if output_idx < self.n_bits:
            bits[output_idx] = n % 2
            output_idx += 1
            n /= 2 
    return bits 

import math 


# limits and target sum derived from empirical stats of some random
# image of a kid on flickr
def mk_encoder(quality = 25,  lowest = -100, highest=60):
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
  
  bits_per_bin = np.zeros_like(quant_values, dtype='int')
  for i,idx in enumerate(sorted_indices):
    
    q = float(quant_values[idx])
    value_range = int(highest/q)*q - int(lowest/q)*q
    if value_range > 0:
      nbits = int(np.ceil(np.log2(float(value_range) / q)))
    else:
      nbits = 0
    bits_per_bin[idx] = int(nbits)
  total_nbits = np.sum(bits_per_bin) 
  encoder = BitEncoder(
    total_nbits, 
    quant, 
    bits_per_bin,
    lowest,
    highest)
  return encoder


def test_random_vectors(encoder, n_iters=1000, verbose=False):
    n_failed = 0 
    n_clipped_below = 0
    n_clipped_above = 0
    for i in xrange(n_iters):
        vec = np.random.randn(encoder.n_bits) > 0
        dct = encoder.encode(vec)
        inverse = scipy.fftpack.dct(dct, type=3, norm='ortho').astype('int')

        
        # clip 
        clipped = inverse.copy()
        bottom = inverse < -128
        clipped[bottom] = -128 
        top = inverse > 127
        clipped[top] = 127 
        n_clipped_below += (0 if np.sum(bottom) == 0 else 1)
        n_clipped_above += (0 if np.sum(top) == 0 else 1)
        dct2 = scipy.fftpack.dct(clipped.astype('float'), type=2, norm='ortho')
        vec2 = encoder.decode(dct2)
        n_failed += (0 if np.all(vec == vec2) else 1)
      
    if verbose:
      print "# clipped below: %d / %d" % (n_clipped_below, n_iters)
      print "# clipped above: %d / %d " % (n_clipped_above, n_iters)
      print "# failed: %d / %d" % (n_failed, n_iters)
    return n_failed, n_clipped_below, n_clipped_above
    
def search(quality=(50,55), lowest=(-300,-40), highest=(19,300), n_tests = 50):
    best_nbits = 0
    best_encoder = None
    best_quality = None
    best_low = None
    best_high = None
    for q in np.arange(quality[0], quality[1],2)[::-1]:
        for l in np.arange(lowest[0], lowest[1], 1)[::-1]:
            last_failed = False 
            h = highest[0]
            while h <= highest[1]:
                en = mk_encoder(q, l, h)
                if en.n_bits > 0:
                    n_failed, _, _ = test_random_vectors(en, n_tests)
                    print "quality = %d, low = %d, high = %d | failed = %d / %d" % (q,l,h,n_failed, n_tests)
                    if n_failed == 0:
                        print "--> nbits = %d" % en.n_bits 
                        last_failed = False
                    else:
                        if last_failed: h += 100
                        last_failed = True
                    if n_failed == 0 and en.n_bits > best_nbits:
                        best_nbits = en.n_bits
                        best_encoder = en
                        best_quality = q
                        best_low = l
                        best_high = h
                    h += 1
            print 
            print
            print "*** BEST SO FAR ***"
            print "low: %d, high: %d, quality: %d" % (best_low, best_high, best_quality)
            print "# bits: %d" % best_encoder.n_bits 
            print
    return best_encoder
