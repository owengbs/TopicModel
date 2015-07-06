#-*- coding:utf-8 -*-
import math, numpy
import CommonWords

"""
@author: macxin
@contact: macxin@tencent.com
@license: GPL
@summary: 
单主题模型算法，参考https://d396qusza40orc.cloudfront.net/textanalytics/lecture_notes/wk2/TM-16-one-topic.pdf
UniTopic算法考虑两个主题分布：目标主题和背景主题
背景主题：主题词来自在线语料库的现代汉语词典词频统计：http://www.cncorpus.org/Resources.aspx
目标主题：待分析的主题模型
随机生成模型：观察样本的每个词是经过对两个主题进行抽样再从每个主题中抽样目标词形成。各个词来自哪个主题视为隐变量Z

求解算法：EM（Expectation-Maximization）
EStep：固定主题词分布，更新Z的预测值
MStep：固定Z，最优化主题词分布

可调整的领域参数：
p_theta_d：目标主题的概率
p_theta_b：背景主题的概率（p_theta_d + p_theta_b = 1）

输出：打印目标主题的top个主题词
'"""

class UniTopicModel():
    def __init__(self):
        self.word_freq = []
        self.word_idx = {}
        self.idx_word = {}
        self.total_count = 0
        self.p_theta_d = 0.5#destination topic
        self.p_theta_b = 0.5#background topic
        self.epsilon = 1e-30

    def _cal_p_z0_g_w(self, wi):
        nominator = self.p_theta_d * self.p_w_g_theta_d[wi]
        denominator = self.p_theta_d * self.p_w_g_theta_d[wi] + self.p_theta_b * self.p_w_g_theta_b[wi]
        if denominator == 0:
            denominator = self.epsilon
        return nominator / denominator
    
    def _cal_p_w_g_theta_d(self, wi):
        nominator = self.word_freq[wi] * self.p_z0_g_w[wi]
        denominator = 0
        for wj in range(self.feature_size):
            denominator += self.word_freq[wj] * self.p_z0_g_w[wj]
        if denominator == 0:
            denominator = self.epsilon
        return nominator / denominator
    
    def likely_hood(self):
        result = 0
        for i in range(self.feature_size):
            logval = self.p_theta_d*self.p_w_g_theta_d[i] + self.p_theta_b*self.p_w_g_theta_b[i]
            result += self.word_freq[i] * math.log( logval  )
        return result
    def estep(self):
        for i in range(self.feature_size):
            self.p_z0_g_w[i] = self._cal_p_z0_g_w(i)
    
    def mstep(self):
        for i in range(self.feature_size):
            self.p_w_g_theta_d[i] = self._cal_p_w_g_theta_d(i)
    
    def initialize(self):
        self.feature_size = len(self.word_idx)
        self.p_w_g_theta_d = numpy.zeros(self.feature_size, float)
        self.p_w_g_theta_b = numpy.zeros(self.feature_size, float)
        self.p_z0_g_w = numpy.zeros(self.feature_size, float)
        for i in range(self.feature_size):
            word = self.idx_word[i]
            self.p_w_g_theta_d[i] = 1.0/self.feature_size
            self.p_w_g_theta_b[i] = 1.0/self.feature_size
            self.p_w_g_theta_b[i] = CommonWords.commonfreq.freqdict[word] if CommonWords.commonfreq.freqdict.has_key(word) else 1.0/self.feature_size
    
    def dump_topic_word(self):
        items = [(i, self.p_w_g_theta_d[i]) for i in range(self.feature_size)]
        items_sorted = sorted(items, cmp = lambda x,y:cmp(x[1], y[1]), reverse = True)
        for i in range(min(20, len(items_sorted))):
            print self.idx_word[items_sorted[i][0]]+" ",
        print '\n'
