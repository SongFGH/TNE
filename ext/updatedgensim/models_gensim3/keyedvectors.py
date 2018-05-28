from gensim.models.word2vec import *
from ext.updatedgensim.models_gensim3.utils_any2vec import _save_word2vec_topic_format
from gensim.models.keyedvectors import *


class TNEBaseKeyedVectors(BaseKeyedVectors):

    def __init__(self, vector_size):
        super(TNEBaseKeyedVectors, self).__init__(vector_size)
        self.vectors_topic = []


class TNEWord2VecKeyedVectors(Word2VecKeyedVectors):

    def save_word2vec_topic_format(self, fname, fvocab=None, binary=False, total_vec=None):
        # from gensim.models.word2vec import save_word2vec_format
        _save_word2vec_topic_format(fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)


KeyedVectors = TNEWord2VecKeyedVectors