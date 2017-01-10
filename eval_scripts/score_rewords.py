from read_write import read_word_vectors,read_word_vectors_orig
from ent_eval import *
import os
import sys
import time
import argparse

default_data = "../xling-entailment/data/monoling_entailment/baroni2012/data_lex_test.tsv"


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--m', action="store", dest="model", required=True)
    parser.add_argument('--d', action="store", dest="datapath", default=default_data)
    opts=parser.parse_args(sys.argv[1:])

    scorer=W2VScorer(datapath=opts.datapath)
    start = time.time()
    vecs=read_word_vectors_orig(opts.model)
    # re_vecs,im_vecs=read_word_vectors_orig(opts.model+".real"),read_word_vectors_orig(opts.model+".imag")
    end = time.time()
    print "elapsed in loading", end-start
    missed,scores = scorer.compute_scores(vecs)
    print "missed",missed
    scorer.get_best_perf(scores)
