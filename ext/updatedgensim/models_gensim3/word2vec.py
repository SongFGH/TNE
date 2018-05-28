from gensim.models.word2vec import *
from ext.updatedgensim.models_gensim3.base_any2vec import *
from ext.updatedgensim.models_gensim3.keyedvectors import TNEWord2VecKeyedVectors
#from lib.updatedgensim.models.keyedvectors import TNEKeyedVectors

"""
import gensim
print(gensim.__version__)

try:
    import pyximport
    from numpy import get_include
    models_dir = os.path.dirname(__file__) or os.getcwd()
    pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
    # import pyximport; pyximport.install()
    from lib.updatedgensim.models.word2vec_inner import train_batch_sg_topic

except ImportError:
    raise Exception("A problem occurred while loading the optimized version")
"""
try:
    import pyximport
    from numpy import get_include
    models_dir = os.path.dirname(__file__) or os.getcwd()
    pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
    from ext.updatedgensim.models_gensim3.word2vec_inner import train_batch_sg_topic

except ImportError:
    print("An error has occurred while loading the optimized version!")


class Word2VecTNE(Word2Vec, TNEBaseWordEmbeddingsModel):
    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):

        self.callbacks = callbacks
        self.load = call_on_class_only

        self.wv = Word2VecKeyedVectors(size)
        self.vocabulary = Word2VecVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=bool(sorted_vocab), null_word=null_word)
        self.trainables = Word2VecTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        super(Word2VecTNE, self).__init__(
            sentences=sentences, workers=workers, vector_size=size, epochs=iter, callbacks=callbacks,
            batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window, seed=seed,
            hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, compute_loss=compute_loss,
            fast_version=FAST_VERSION)
    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):

        self.callbacks = callbacks
        self.load = call_on_class_only

        self.wv = TNEWord2VecKeyedVectors(size)
        self.vocabulary = Word2VecVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=bool(sorted_vocab), null_word=null_word)
        self.trainables = TNEWord2VecTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        TNEBaseWordEmbeddingsModel.__init__(self,
            sentences=sentences, workers=workers, vector_size=size, epochs=iter, callbacks=callbacks,
            batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window, seed=seed,
            hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, compute_loss=compute_loss,
            fast_version=FAST_VERSION)


    def _do_train_job_topic(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """

        work, neu1 = inits

        tally = 0
        if self.sg:
            tally += train_batch_sg_topic(self, sentences, alpha, work, self.compute_loss)
        else:
            raise ValueError("It has not been implemented for CBOW!")
            #tally += train_batch_cbow(self, sentences, alpha, work, neu1, self.compute_loss)
        return tally, self._raw_word_count(sentences)

    def train_topic(self, number_of_topics, sentences, total_examples=None, total_words=None,
                    epochs=None, start_alpha=None, end_alpha=None, word_count=0,
                    queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=()):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, and accurate
        progress-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) **MUST** be provided (if the corpus is the same as was provided to
        :meth:`~gensim.models.word2vec.Word2Vec.build_vocab()`, the count of examples in that corpus
        will be available in the model's :attr:`corpus_count` property).

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case,
        where :meth:`~gensim.models.word2vec.Word2Vec.train()` is only called once,
        the model's cached `iter` value should be supplied as `epochs` value.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float
            Initial learning rate.
        end_alpha : float
            Final learning rate. Drops linearly from `start_alpha`.
        word_count : int
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float
            Seconds to wait before reporting progress.
        compute_loss: bool
            If True, computes and stores loss value which can be retrieved using `model.get_latest_training_loss()`.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        >>> from gensim.models import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = Word2Vec(min_count=1)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

        """
        self.trainables.reset_topic_weights(number_of_topic=number_of_topics, hs=self.negative, negative=self.negative, wv=self.wv)

        total_examples = self.corpus_count
        epochs = self.iter

        return super(Word2VecTNE, self).train_topic(number_of_topics,
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)


class TNEWord2VecTrainables(Word2VecTrainables):

    def reset_topic_weights(self, number_of_topic, hs, negative, wv):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        wv.vectors_topics = empty((number_of_topic, wv.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        #for i in xrange(len(wv.vocab)):
        for i in xrange(number_of_topic):
            # construct deterministic seed from word AND seed argument
            wv.vectors_topics[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)

        wv.vectors_norm_topic = None

        self.vectors_lockf_topic = ones(number_of_topic, dtype=REAL)  # zeros suppress learning


class CombineSentences(object):

    def __init__(self, node_filename, topic_filename):
        self.topic_filename = topic_filename
        self.node_filename = node_filename

    def __iter__(self):
        for nodes, topics in zip(Text8Corpus(self.node_filename), Text8Corpus(self.topic_filename)):
            yield [(u, int(v)) for (u, v) in zip(nodes, topics)]
