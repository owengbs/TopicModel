#-*- coding:utf-8 -*-
"""
common_word_filepath set to path of common word dictionary file
"""

common_word_filepath = "./data/wordfreq.txt"

class CommonWordFrequency():
    def __init__(self, freq_file):
        self.freqdict = {}
        self.load_common_word_frequency(freq_file)
    def load_common_word_frequency(self, freq_file):
        fd = open(freq_file, 'r+')
        for line in fd.readlines():
            segs = line.split()
            if len(segs) == 3:
                self.freqdict[segs[0]] = float(segs[2])
                
commonfreq = CommonWordFrequency(common_word_filepath)