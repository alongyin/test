import jieba
from gensim.models import LdaModel
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
import codecs

#读取停用词的列表
def getStopWordsList(filePath):
    stopWords = [line.strip() for line in open(filePath,'r',encoding = 'utf-8').readlines()] 
    return stopWords
#进行分词、过滤停用词处理
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopWordsList = getStopWordsList('stopwords/1893.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopWordsList:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


#进行预处理的模块
def getPreProcssWord():
    inputs = open('input/input.txt','r',encoding='utf-8')
    outputs = open('outputs/output.txt','w',encoding='utf-8')
    for line in inputs:
        line_seg = seg_sentence(line)
        outputs.write(line_seg + "\n")
    outputs.close()
    inputs.close()

    
#进行LDA主题训练模块
train = []
def trainModel():
    fp = codecs.open('outputs/output.txt','r',encoding='utf8')
    for line in fp:
        line =  line.split()
        train.append([ w for w  in line])
    dictionary = corpora.Dictionary(train)
    corpus = [ dictionary.doc2bow(text) for text in train]

    #进行LDA训练
    lda = LdaModel(corpus=corpus,id2word=dictionary,num_topics=5,passes=20)
    model_path = datapath("model")
    lda.save(model_path)


def useModel():
    model_path = datapath("model")
    print("model_path:",model_path)
    lda = LdaModel.load(model_path)

    #主题的分布概率
    for topic in lda.print_topics(num_words=100):
        termNumber = topic[0]
        print(topic[0],":",sep='')
        list0fTerms = topic[1].split("+")
        for term in list0fTerms:
            listItems = term.split('*')
            print(' ',listItems[1],'(',listItems[0],')',sep='')
    
    #get_term_topics
    term_topics_list = lda.get_term_topics("不同")
    print(term_topics_list)
    #get_document_topics


if __name__ == "__main__":
    stage = 3 # 1. 进行数据读取、分词、过滤并最终存储；2. 进行LDA话题训练和评估;3. 使用model
    if stage == 1:
        getPreProcssWord()
    if stage == 2:
        trainModel()
    if stage == 3:
        useModel()