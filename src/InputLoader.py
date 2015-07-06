#-*- coding:utf-8 -*-
import re, os, numpy

class InputLoader():
    def __init__(self):
        pass
    """
    input file:text segments 
    
    return (word_idx, idx_word, word_freq, total_count)
    """
    def trimchinese(self, line):
        re_chinese = re.compile(u'[^a-zA-Z\u4e00-\u9fa5]')
        line = re_chinese.sub(' ', line.decode('utf-8').rstrip())
        return line.encode("utf8")
    def load_x_vector_by_file(self, filename):
        total_count = 0
        word_idx, idx_word, word_freq = {}, {}, []
        fd = open(filename, 'r+')
        
        for line in fd.readlines():
            segs = self.trimchinese(line).split()
            total_count += len(segs)
            for word in segs:
                if not word_idx.has_key(word):
                    word_idx[word] = len(word_idx)
                    idx_word[word_idx[word]] = word
                    word_freq.append(1)
                else:
                    word_freq[word_idx[word]] += 1
        return (word_idx, idx_word, word_freq, total_count)
    
    def load_cluster_vector_by_dir(self, dirpath):
        word_idx, idx_word, docsize, featuresize = {}, {}, 0, 0
        for filename in os.listdir(dirpath):
#            print filename
            docsize += 1
            tmp_dataset_file_name = os.path.join(dirpath, filename)
            fd = open(tmp_dataset_file_name, 'r')
            for line in fd.readlines():
                segs = self.trimchinese(line).split()
                for word in segs:
                    if not word_idx.has_key(word):
                        word_idx[word] = len(word_idx)
                        idx_word[word_idx[word]] = word
        featuresize = len(word_idx)
        count_word = numpy.zeros((docsize, featuresize), int)
        dirlist= os.listdir(dirpath)
        for docidx in range(len(dirlist)): 
            fd = open(os.path.join(dirpath, dirlist[docidx]), 'r')
            for line in fd.readlines():
                segs = self.trimchinese(line).split()
                for word in segs:
                    count_word[docidx][word_idx[word]] += 1
#        print word_idx
        return (word_idx, idx_word, count_word)
