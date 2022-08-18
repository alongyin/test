#中文版处理的word2vector

from gensim import models

file_path = "./data/1.txt"

#获取输入的信息，并进行标准化处理
class getFormatInput(object):
    def __init__(self,file_path):
        self.file_path = file_path
    
    def __iter__(self):
        for line in open(file_path):
            # split 当只有1个时候，返回单个字符串，多个则返回多个字符串
            words = line.split(" ")
            result_word = []
            for word in words:
                if word and word != '\n':
                    result_word.append(word)
                yield result_word
formatInput = getFormatInput(file_path)
model = models.Word2Vec(formatInput,workers=20,min_count=5,vector_size=200)
molde.save("./zh_word2vec")







