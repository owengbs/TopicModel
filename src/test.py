#-*- coding:utf-8 -*-
from scipy.optimize import *
import theano.tensor as T
import math, numpy, sys, re
import CommonWords
from numpy.core.test_rational import denominator
from InputLoader import *

word_freq = []
word_idx = {}
idx_word = {}
total_count = 0
p_theta_d = 0.5
p_theta_b = 0.5
epsilon = 1e-30


samplefile = sys.argv[1]
loader = InputLoader()
(word_idx, idx_word, word_freq, total_count) = loader.loadXVectorByFile(samplefile)

feature_size = len(word_freq)
p_w_g_theta_d = numpy.zeros(feature_size, float)
p_w_g_theta_b = numpy.zeros(feature_size, float)

p_z0_g_w = numpy.zeros(feature_size, float)

def _cal_p_z0_g_w(wi):
    nominator = p_theta_d * p_w_g_theta_d[wi]
    denominator = p_theta_d * p_w_g_theta_d[wi] + p_theta_b * p_w_g_theta_b[wi]
    if denominator == 0:
        denominator = epsilon
    return nominator / denominator

def _cal_p_w_g_theta_d(wi):
    nominator = word_freq[wi] * p_z0_g_w[wi]
    denominator = 0
    for wj in range(feature_size):
        denominator += word_freq[wj] * p_z0_g_w[wj]
    if denominator == 0:
        denominator = epsilon
    return nominator / denominator

def LogLikelyHood():
    result = 0
    for i in range(feature_size):
        logval = p_theta_d*p_w_g_theta_d[i] + p_theta_b*p_w_g_theta_b[i]
        result += word_freq[i] * math.log( logval  )
    return result
def EStep():
    for i in range(feature_size):
        p_z0_g_w[i] = _cal_p_z0_g_w(i)

def MStep():
    for i in range(feature_size):
        p_w_g_theta_d[i] = _cal_p_w_g_theta_d(i)

def initialize():
    for i in range(feature_size):
        word = idx_word[i]
        p_w_g_theta_d[i] = 1.0/feature_size
        p_w_g_theta_b[i] = 1.0/feature_size
        p_w_g_theta_b[i] = CommonWords.commonfreq.freqdict[word] if CommonWords.commonfreq.freqdict.has_key(word) else 1.0/feature_size

def dumpTopWord():
    items = [(i, p_w_g_theta_d[i]) for i in range(feature_size)]
    items_sorted = sorted(items, cmp = lambda x,y:cmp(x[1], y[1]), reverse = True)
    for i in range(min(20, len(items_sorted))):
        print idx_word[items_sorted[i][0]]+" ",
    print '\n'
iters = 0
max_iter = 100
# STEP-1: Initialization
initialize()
while iters < max_iter: 
    iters += 1
    EStep()
    MStep()
#     print LogLikelyHood()
print samplefile
dumpTopWord()