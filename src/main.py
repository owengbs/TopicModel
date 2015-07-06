#-*- coding:utf-8 -*-
import sys, InputLoader, UniTopicModel, UniGenerativeModelClustering
def usage():
    print "usage:python main.py cmd[topic/cluster] samplefile"
if len(sys.argv) < 3:
    usage()
    exit()
cmd = sys.argv[1]
samplefile = sys.argv[2]
loader = InputLoader.InputLoader()
model = None
if cmd == "topic":
    model = UniTopicModel.UniTopicModel()
    (model.word_idx, model.idx_word, model.word_freq, model.total_count) = loader.load_x_vector_by_file(samplefile)
    model.initialize()
else:
    model = UniGenerativeModelClustering.UniGenerativeModelClustering()
    (model.word_idx, model.idx_word, model.c_w_g_d) = loader.load_cluster_vector_by_dir(samplefile)
    model.initialize()
    
iters = 0
max_iter = 10
while iters < max_iter: 
    iters += 1
    model.estep()
    model.mstep()
    print model.likely_hood()
model.dumpClusters()