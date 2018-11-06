import numpy as np
import os, pickle
from config import Config

vocab_tags = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

class Dataset(object):
    """
    Class that iterates over source data
    __iter__ method yields a tuple(words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is applied
    
    Example:
        data = Dataset(file)
        for sentence, tags in data:
            pass
    """
    def __init__(self, words_filename, tags_filename, max_iter=None):
        """
        Args:
            words_filename: path to file
            tags_filename: path to file
            # processing_word:(optional) function that takes a word as input
            # processing_tag:(optional) function that takes a tag as input
            max_iter:(optional) max number of sentences to yield
        """
        self.words_filename = words_filename
        self.tags_filename = tags_filename
        # self.processing_word = processing_word
        # self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    
    def __iter__(self):
        niter = 0
        words_file = open(self.words_filename, mode="r")
        tags_file = open(self.tags_filename, mode="r")

        for word_line, tag_line in zip(words_file, tags_file):
            sentences, tags = [],[]
            word_line = word_line.strip()
            tag_line = tag_line.strip()
            if word_line.startswith("-DOCSTART-"):
                if len(sentences) != 0:
                    niter += 1
                    if self.max_iter is not None and niter > self.max_iter:
                        break
                    # yield sentences, tags
                    # sentences, tags = [], []
            else:
                sentence = word_line.split(" ")
                tag = tag_line.split(" ")

                sentences += sentence
                tags += tag
                yield sentence, tags
        

        words_file.close()
        tags_file.close()

    
    def __len__(self):
        if self.length is None:
            self.length=0
            for _ in self:
                self.length += 1
        return self. length



def vocab_build(data, min_count, vocab_path):
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id



def sentence2id(seq, vocab_words):
    new_seq = list()
    for char in seq:
        if char in vocab_words:
            new_seq.append(vocab_words[char])
        else:
            new_seq.append(vocab_words['<UNK>'])
    
    return new_seq


def tag2id(seq, vocab_tags):
    return list(map(lambda tag: vocab_tags[tag], seq))


def minibatches(data, vocab_tags, vocab_words,  minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size : (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        x = sentence2id(x, vocab_words)
        y = tag2id(y, vocab_tags)
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]
    
    if len(x_batch) !=0:
        yield x_batch, y_batch


def _pad_sequence(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        sequence_length += [len(seq)]
        seq = seq + [pad_tok] * (max_length-len(seq))
        sequence_padded += [seq]
    
    return sequence_padded, sequence_length


def pad_sequence(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequence(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def get_chunk_type(tok, idx_to_tag):
    # copy from guillaumegenthial / sequence_tagging
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    # copy from guillaumegenthial / sequence_tagging
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks