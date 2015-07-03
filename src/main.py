#-*- coding:utf-8 -*-
import sys, InputLoader, UniTopic
def usage():
    print "usage:python main.py samplefile"
if len(sys.argv) < 2:
    usage()
    exit()
samplefile = sys.argv[1]

loader = InputLoader.InputLoader()
unitopic = UniTopic.UniTopic()
(unitopic.word_idx, unitopic.idx_word, unitopic.word_freq, unitopic.total_count) = loader.load_x_vector_by_file(samplefile)
unitopic.initialize()

iters = 0
max_iter = 100
while iters < max_iter: 
    iters += 1
    unitopic.estep()
    unitopic.mstep()
print samplefile
unitopic.dump_topic_word()