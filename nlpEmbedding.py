from gensim import corpora
texts = [['human', 'interface', 'computer'],
['survey', 'user', 'computer', 'system', 'response', 'time'],
['eps', 'user', 'interface', 'system'],
['system', 'human', 'system', 'eps'],
['user', 'response', 'time'],
['trees'],
['graph', 'trees'],
['graph', 'minors', 'trees'],
['graph', 'minors', 'survey']]


#构建词袋的索引
dictionary = corpora.Dictionary(texts)
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]

#主题向量的变换
from gensim import models
tfidf = models.TfidfModel(corpus)
doc_bow = [(0,1),(1,1)]
print(tfidf[doc_bow])
