#!/usr/bin/env python3
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
import pickle
from collections import defaultdict
import numpy as np
import math
import os

def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    return precook(test, n, True)

class CiderScorer(object):
    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), f"refs/test mismatch! {len(self.crefs)}<>{len(self.ctest)}"
        return len(self.crefs)

    def __iadd__(self, other):
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, _) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency.get(ngram, 0.0)))
                n = len(ngram) - 1
                if n >= self.n:
                    continue
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, _) in vec_hyp[n].items():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n].get(ngram, 0.0)
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                assert not math.isnan(val[n])
            return val

        self.ref_len = np.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, df_mode, pfile_path, option=None, verbose=0):
        with open(pfile_path, 'rb') as f:
            self.document_frequency = pickle.load(f)
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=1, df="coco-val-df"):
        """
        Initialize the CIDEr scoring function
        : param n (int): n-gram size
        : param df (string): specifies where to get the IDF values from
                    takes values 'corpus', 'coco-train'
        : return: None
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        self._df = df

    def compute_score(self, gts, res, pfile_path):
        """
        Main function to compute CIDEr score
        : param  gts (dict) : {image:tokenized reference sentence}
        : param res (dict)  : {image:tokenized candidate sentence}
        : return: cider (float) : computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n)

        for res_id in res:

            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score(self._df, pfile_path= pfile_path)

        return score, scores

    def method(self):
        return "CIDEr"
    