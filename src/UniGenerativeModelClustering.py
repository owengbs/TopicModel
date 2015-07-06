#-*- coding:utf-8 -*-
import numpy, math, random
class UniGenerativeModelClustering():
    def __init__(self, k = 2):
        self.document_size = 0
        self.kclusters = k
        self.feature_size = 0
        self.word_idx = {}
        self.idx_word = {}
        self.epsilon = 1e-30
        
    def initialize(self):
        self.feature_size = len(self.word_idx)
        self.document_size = len(self.c_w_g_d)
        self.p_w_g_theta = numpy.zeros((self.kclusters, self.feature_size), float)
        self.p_avg_w_g_theta = numpy.zeros(self.feature_size, float)
        self.p_zk_g_d = numpy.zeros((self.document_size, self.kclusters), float)
        self.p_theta = numpy.zeros(self.kclusters, float)
        
        #    self.p_avg_w_g_theta[featureidx] = 1.0/self.kclusters
#            print "avgp[%d]=%f" % (featureidx, self.p_avg_w_g_theta[featureidx])
        for clusteridx in range(self.kclusters):
            for featureidx in range(self.feature_size):
                if featureidx == self.feature_size - 1:
                    self.p_w_g_theta[clusteridx][featureidx] = 1 - sum(self.p_w_g_theta[clusteridx])
                else:
                    left = 1.0-sum(self.p_w_g_theta[clusteridx])
                    tmprandom = random.random() * left
                    self.p_w_g_theta[clusteridx][featureidx] = tmprandom
        self._update_p_avg_w_g_theta()
        for clusteridx in range(self.kclusters):
            self.p_theta[clusteridx] = 1.0/self.kclusters
            for docidx in range(self.document_size):
                self.p_zk_g_d[docidx][clusteridx] = 1.0/self.kclusters
    
    def _update_p_avg_w_g_theta(self ):
        for featureidx in range(self.feature_size):
            tmpsum = 0.0
            for clusteridx in range(self.kclusters):
                tmpsum += self.p_w_g_theta[clusteridx][featureidx]
            self.p_avg_w_g_theta[featureidx] = tmpsum / self.kclusters
#            print "update avgp[%d]=%0.6f" % (featureidx, self.p_avg_w_g_theta[featureidx])
    def _cal_p_zk_g_d(self, docidx, clusteridx):
#        print "featuresize=%d" % self.feature_size
        nominator = self.p_theta[clusteridx]
        for featureidx in range(self.feature_size):
#            print "(p[%d|%d]/avgp[%d]=%f/%f)^%d *" % (featureidx, clusteridx, featureidx, self.p_w_g_theta[clusteridx][featureidx], self.p_avg_w_g_theta[featureidx], self.c_w_g_d[docidx][featureidx]),
            nominator *= (self.p_w_g_theta[clusteridx][featureidx]/self.p_avg_w_g_theta[featureidx]) ** self.c_w_g_d[docidx][featureidx]
#        print ''

        denominator = 0
        for subclusteridx in range(self.kclusters):
            subnominator = self.p_theta[subclusteridx]
            for featureidx in range(self.feature_size):
                subnominator *= (self.p_w_g_theta[subclusteridx][featureidx]/self.p_avg_w_g_theta[featureidx]) ** self.c_w_g_d[docidx][featureidx]
#                print "1:%f 2:%f 1/2:%f c:%d sub:%f" % (self.p_w_g_theta[subclusteridx][featureidx],self.p_avg_w_g_theta[featureidx],self.p_w_g_theta[subclusteridx][featureidx]/self.p_avg_w_g_theta[featureidx],self.c_w_g_d[docidx][featureidx],subnominator)
            denominator += subnominator
            
        denominator = self.epsilon if denominator == 0 else denominator
#        print "doc:%d cluster:%d nominator=%f denominator=%f" % (docidx, clusteridx, nominator, denominator)
        return nominator / denominator
                
    def _cal_p_theta(self, clusteridx):
        self.p_theta[clusteridx] = 0
        for docidx in range(self.document_size):
            self.p_theta[clusteridx] += self.p_zk_g_d[docidx][clusteridx]
        self.p_theta[clusteridx] /= self.document_size
#        print "p_theta[%d]=%f" % (clusteridx,self.p_theta[clusteridx])

    def _cal_p_w_g_theta(self, clusteridx):
#            print self.feature_size
            for featureidx in range(self.feature_size):
                self.p_w_g_theta[clusteridx][featureidx] = 0
                for docidx in range(self.document_size):
                    self.p_w_g_theta[clusteridx][featureidx] += self.c_w_g_d[docidx][featureidx]*self.p_zk_g_d[docidx][clusteridx]
#                print "%d %d %d %0.6f" % (clusteridx, featureidx, self.c_w_g_d[clusteridx][featureidx], self.p_w_g_theta[clusteridx][featureidx])
            tmpsum = sum(self.p_w_g_theta[clusteridx])
#            print tmpsum
            self.p_w_g_theta[clusteridx] = [self.p_w_g_theta[clusteridx][i] / tmpsum for i in range(self.feature_size)]
#            for featureidx in range(self.feature_size):
#                print "p_w_g_theta[%d|%d]=%f" % (featureidx, clusteridx, self.p_w_g_theta[clusteridx][featureidx])
#            print self.p_avg_w_g_theta
    def estep(self):
        for docidx in range(self.document_size):
            for clusteridx in range(self.kclusters):
                self.p_zk_g_d[docidx][clusteridx] = self._cal_p_zk_g_d(docidx, clusteridx)
#                print "p_zk_g_d[%d|%d]=%f" % (clusteridx, docidx, self.p_zk_g_d[docidx][clusteridx])

    def mstep(self):
        for clusteridx in range(self.kclusters):
            """
                update p(theta)
            """
            self._cal_p_theta(clusteridx)
            """
                update p(w|theta)
            """
            self._cal_p_w_g_theta(clusteridx)
        self._update_p_avg_w_g_theta()

    def likely_hood(self):
        result = 1
        for docidx in range(self.document_size):
            retsult_doc = 0.0
            for clusteridx in range(self.kclusters):
                dochood = self.p_theta[clusteridx]
#                print "dochood before:%f" % dochood
                for featureidx in range(self.feature_size):
                    dochood *= (self.p_w_g_theta[clusteridx][featureidx] ** self.c_w_g_d[docidx][featureidx])
                    tmp = self.p_w_g_theta[clusteridx][featureidx] ** self.c_w_g_d[docidx][featureidx]
#                    if tmp != 1:
#                        print "docid=%d cluster=%d tmp=%f theta=%f count=%f" % (docidx, clusteridx, tmp, self.p_w_g_theta[clusteridx][featureidx], self.c_w_g_d[docidx][featureidx])
#                print "dochood:%f" % dochood
                retsult_doc += dochood
#            print retsult_doc
            result *= retsult_doc
        return result
    def dumpClusters(self):
        for docidx in range(self.document_size):
            maxclusteridx = 0
            maxz = 0
            for clusteridx in range(self.kclusters):
                if self.p_zk_g_d[docidx][clusteridx] > maxz:
                    maxz = self.p_zk_g_d[docidx][clusteridx]
                    maxclusteridx = clusteridx
            print "doc%d:%d" % (docidx, maxclusteridx)