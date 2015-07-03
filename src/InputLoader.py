#-*- coding:utf-8 -*-
import re

class InputLoader():
    def __init__(self):
        pass
    """
    input file:text segments 
    
    return (word_idx, idx_word, word_freq, total_count)
    """
    def load_x_vector_by_file(self, filename):
        total_count = 0
        word_idx, idx_word, word_freq = {}, {}, []
        fd = open(filename, 'r+')
        
        for line in fd.readlines():
            re_chinese = re.compile(u'[^a-zA-Z\u4e00-\u9fa5]')
            line = re_chinese.sub(' ', line.decode('utf-8').rstrip())
    #         line = re.sub("[\s+\.\!\/_,$%^*(+\"\'\]\[\:]+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), " ".decode("utf8"),line.decode("utf8"))  
            segs = line.encode("utf8").split()
            total_count += len(segs)
            for word in segs:
                if not word_idx.has_key(word):
                    word_idx[word] = len(word_idx)
                    idx_word[word_idx[word]] = word
                    word_freq.append(1)
                else:
                    word_freq[word_idx[word]] += 1
        return (word_idx, idx_word, word_freq, total_count)