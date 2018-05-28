from gensim.utils import *
from numpy import zeros as REAL


def _save_word2vec_topic_format(fname, vocab, vectors_topic, fvocab=None, binary=False, total_vec=None):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in
    vocab : dict
        The vocabulary of words
    vectors : numpy.array
        The vectors to be stored
    fvocab : str
        Optional file path used to save the vocabulary
    binary : bool
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec :  int
        Optional parameter to explicitly specify total no. of vectors
        (in case word vectors are appended with document vectors afterwards)

    """
    if not (vocab or vectors_topic):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors_topic.shape[1]
    if fvocab is not None:
        logger.info("storing topic vocabulary in %s", fvocab)
        with smart_open(fvocab, 'wb') as vout:
            for word, vocab_ in sorted(iteritems(vocab), key=lambda item: -item[1].count):
                vout.write(to_utf8("%s %s\n" % (word, vocab_.count)))
    logger.info("storing %sx%s projection weights into %s", total_vec, vector_size, fname)
    assert (len(vocab), vector_size) == vectors_topic.shape
    with smart_open(fname, 'wb') as fout:
        fout.write(to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, vocab_ in sorted(iteritems(vocab), key=lambda item: -item[1].count):
            row = vectors_topic[vocab_.index]
            if binary:
                row = row.astype(REAL)
                fout.write(to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))