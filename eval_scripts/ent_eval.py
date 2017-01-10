import sys
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import time
from gzip import GzipFile
from sklearn.metrics import *
import numpy as np
# from word2gauss import GaussianEmbedding
# from word2gauss import iter_pairs
# from word2gauss.words import Vocabulary, vocab_from_counts_file

default_data = "../xling-entailment/data/monoling_entailment/baroni2012/data_lex_test.tsv"





# def get_pred(scores, thres):
#     pred = []
#     for score in scores:
#         if score is None:
#             pred.append(None)
#             continue
#         pred.append(True if score > thres else False) # if KL div is large predict true
#     return pred






def split_scores(scores, gold):
    pos_sc = []
    neg_sc = []
    for i, score in enumerate(scores):
        if gold[i]:
            pos_sc.append(score)
        else:
            neg_sc.append(score)
    print "min", min(pos_sc), "max", max(pos_sc), "mean", np.mean(pos_sc), "std", np.std(pos_sc), len(pos_sc)
    print "min", min(neg_sc), "max", max(neg_sc), "mean", np.mean(neg_sc), "std", np.std(neg_sc), len(neg_sc)

class AbstractScorer(object):
    def __init__(self,datapath = default_data):
        data = [line.strip().split() for line in open(datapath)]
        self.data = [(d[0], d[1], d[2]) for d in data]
        self.gold = [True if d[2] == "True" else False for d in data]

    def evaluate(self, pred):
        missed = 0
        f1 = f1_score(self.gold, pred, average='binary')   # average='macro'
        accuracy = accuracy_score(self.gold, pred)
        return f1, accuracy

    def compute_scores(self,model):
        raise NotImplementedError

    def get_pred(self, scores, thres):
        pred = []
        for score in scores:
            if score is None:
                pred.append(None)
                continue
            pred.append(True if score > thres else False) # if KL div is large predict true
        return pred

    def get_best_perf(self,scores):
        fscores, accs = [], []
        # print "min(scores), max(scores)",min(scores), max(scores)
        thres_grid = np.linspace(min(scores), max(scores), 1000)  # [-25]
        best_f1 = (0, None)  # also keep threshold
        best_acc = (0, None)

        for thres in thres_grid:
            pred = self.get_pred(scores, thres)
            fscore, acc = self.evaluate(pred)
            # print pred.count(True),pred.count(False),pred.count(None),fscore,"acc:",acc
            fscores.append(fscore)
            accs.append(acc)
            if fscore > best_f1[0]: best_f1 = (fscore, thres)
            if acc > best_acc[0]: best_acc = (acc, thres)
        assert max(fscores) == best_f1[0]
        assert max(accs) == best_acc[0]
        print "best f1", best_f1, "best acc", best_acc

class EntScorer(AbstractScorer):
    def __init__(self, iw, wi, ic, ci, datapath=default_data):
        super(EntScorer, self).__init__()
        self.iw, self.wi, self.ic, self.ci = iw, wi, ic, ci

    def compute_scores(self, model):
        # print model
        scores = []
        missed = 0
        for d in self.data:
            w1, w2 = d[0].lower(), d[1].lower()
            w1_idx, w2_idx= self.wi[w1],self.wi[w2]
            if w1 not in self.wi or w2 not in self.wi:
                missed+=1
                scores.append(0)
                continue
            # score = model.predict([w1_idx, w2_idx])
            score = model.predict(np.asarray([[w1_idx, w2_idx]]))
            # print w1,w2,score
            scores.append(score[0])
        assert len(self.data) == len(scores)
        return missed, scores


class ComplexScorer(AbstractScorer):
    def compute_scores(self, vecs):
        re_vecs, im_vecs = vecs[0],vecs[1]
        scores = []
        missed = 0
        for d in self.data:
            w1, w2 = d[0].lower(), d[1].lower()
            # w2, w1 = w1, w2  # reversing hurts so yay!
            if w1 not in re_vecs or w2 not in re_vecs:
                missed+=1
                scores.append(0)
                continue
            # score = model.predict([w1_idx, w2_idx])
            score = np.dot(re_vecs[w1],re_vecs[w2]) + np.dot(im_vecs[w1],im_vecs[w2]) \
                    + np.dot(re_vecs[w1],im_vecs[w2]) - np.dot(re_vecs[w2],im_vecs[w1])
            # print w1,w2,score
            scores.append(score)
        assert len(self.data) == len(scores)
        return missed, scores

class W2VScorer(AbstractScorer):
    def compute_scores(self, vecs):
        scores = []
        missed = 0
        for d in self.data:
            w1, w2 = d[0].lower(), d[1].lower()
            # w2, w1 = w1, w2  # reversing hurts so yay!
            if w1 not in vecs or w2 not in vecs:
                missed+=1
                scores.append(0)
                continue
            # score = model.predict([w1_idx, w2_idx])
            score = np.dot(vecs[w1],vecs[w2])
            # print w1,w2,score
            scores.append(score)
        assert len(self.data) == len(scores)
        return missed, scores

class XComplexScorer(AbstractScorer):
    def compute_scores(self, en_vecs, fr_vecs):
        en_re, en_im = en_vecs[0],en_vecs[1]
        fr_re, fr_im = fr_vecs[0],fr_vecs[1]
        scores = []
        missed = 0
        for d in self.data:
            w_fr, w_en = d[0].lower(), d[1].lower()
            # w2, w1 = w1, w2  # reversing hurts so yay!
            if w_fr not in fr_re or w_en not in en_re:
                missed+=1
                scores.append(0)
                continue
            # score = model.predict([w1_idx, w2_idx])
            score = np.dot(fr_re[w_fr],en_re[w_en]) + np.dot(fr_im[w_fr],en_im[w_en]) \
                    + np.dot(fr_re[w_fr],en_im[w_en]) - np.dot(en_re[w_en],fr_im[w_fr])
            # print w1,w2,score
            scores.append(score)
        assert len(self.data) == len(scores)
        return missed, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--m', action="store", dest="model", required=True)
    parser.add_argument('--v', action="store", dest="vocab", default="enwiki.dict.new2")
    parser.add_argument('--d', action="store", dest="datapath", default=default_data)
    parser.add_argument('--thres', action="store", dest="thres", type=float)
    parser.add_argument('--minfreq', action="store", type=int, dest="minfreq", default=100)

    opts = parser.parse_args(sys.argv[1:])
    logging.info("test data: %s", opts.datapath)
    # print vars(opts)
    # sys.exit(0)
