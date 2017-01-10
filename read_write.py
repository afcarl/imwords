import sys
import gzip
import numpy
import math

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):    
  re_word_vecs,im_word_vecs = {},{}
  if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  else: file_object = open(filename, 'r')

  for line_num, line in enumerate(file_object):
    if line_num==5: break
    if len(line.split())==2: continue
    line = line.strip().lower()
    word = line.split()[0]
    DIM = (len(line.split())-1) / 2
    re_word_vecs[word] = numpy.zeros(DIM, dtype=float)
    im_word_vecs[word] = numpy.zeros(DIM, dtype=float)
    tmp = numpy.zeros(2*DIM, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      tmp[index] = float(vec_val)
    re_word_vecs[word]=tmp[:DIM]
    im_word_vecs[word]=tmp[DIM:]
    # ''' normalize weight vector '''
    # word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)        
    print word, re_word_vecs[word],im_word_vecs[word], len(re_word_vecs[word]),len(im_word_vecs[word])
  sys.stderr.write("Vectors read from: "+filename+" \n")
  # return word_vecs
if __name__=="__main__":
  read_word_vectors(sys.argv[1])
  
