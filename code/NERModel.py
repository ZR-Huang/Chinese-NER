from __future__ import absolute_import, division, print_function

import tensorflow as tf
import math
import collections
import numpy as np
import random

class Dataset():

    def __init__(self, trainset_file, trainlabel_file):
        self.trainset_file = trainset_file
        self.trainlabel_file = trainlabel_file
        self.trainset = list()
        self.trainlabel = list()
        self.vocabulary = list()
        # self.n_words = 0   # the top n common words
        self.dictionary = dict() # map of words(strings) to their codes(integers)
        self.reversed_dictionary = dict()
        self.data = list()      # list of codes (integers from 0 to vocabulary_size-1).
        self.count = [['UNK', -1]] # map of words(strings) to count of occurrences
        self.data_index = 0 # for generate the batch

        self.sentence_length = list() # record the length of sentence



    def load_trainset(self):
        with open(self.trainset_file, mode="r") as f:
            for row in f.readlines():
                self.trainset.append(row.strip().split(sep=' '))

        with open(self.trainlabel_file, mode="r") as f:
            for row in f.readlines():
                self.trainlabel.append(row.strip().split(sep=' '))

        #return (trainset, trainlabel)


    def build_vocabulary(self):
        for sentence in self.trainset:
            self.vocabulary.append(sentence)
            self.sentence_length.append(sentence.__len__())
        # return vocabulary

    def build_dataset(self, n_words):
        """Process raw inputs into a dataset."""
        # count = [['UNK', -1]]
        self.count.extend(collections.Counter(self.vocabulary).most_common(n_words - 1))

        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        # data = list()
        unk_count = 0
        for word in self.vocabulary:
            index = self.dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        # return data, count, dictionary, reversed_dictionary


    def generate_batch(self, batch_size, num_skips, skip_window):
      # global data_index
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1  # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
      if self.data_index + span > len(self.data):
        self.data_index = 0
      buffer.extend(self.data[self.data_index:self.data_index + span])
      self.data_index += span
      for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[context_word]
        if self.data_index == len(self.data):
          buffer.extend(self.data[0:span])
          self.data_index = span
        else:
          buffer.append(self.data[self.data_index])
          self.data_index += 1
      # Backtrack a little bit to avoid skipping words in the end of a batch
      self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
      return batch, labels


class NERModel():

    def __init__(self):
        self.batch_size = 128
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 1  # How many words to consider left and right.
        self.num_skips = 2  # How many times to reuse an input to generate a label.
        self.num_sampled = 64  # Number of negative examples to sample.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.train_file = "../data/source_data.txt"
        self.train_label_file = "../data/source_label.txt"
        self.vocabulary_size = 4220


    def run_graph(self):
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)


        with tf.name_scope("embedding"):
            embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)


        with tf.name_scope('embedding_weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [self.vocabulary_size, self.embedding_size],
                    stddev=1.0 / math.sqrt(self.embedding_size)))
        with tf.name_scope('embedding_biases'):
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))


        with tf.name_scope('embedding_loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size))


        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()
        num_steps = 100001

        with tf.Session() as session:

            # We must initialize all variables before we use them.
            init.run()
            dataset = Dataset(self.train_file, self.train_label_file)
            dataset.load_trainset()
            dataset.build_vocabulary()
            dataset.build_dataset(dataset.vocabulary.__len__())
            print('Initialized')

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = dataset.generate_batch(self.batch_size, self.num_skips,
                                                                    self.skip_window)

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}


                _, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict=feed_dict)
                average_loss += loss_val


                # if step % 2000 == 0:
                #     if step > 0:
                #         average_loss /= 2000
                #     # The average loss is an estimate of the loss over the last 2000 batches.
                #     print('Average loss at step ', step, ': ', average_loss)
                #     average_loss = 0
                #
                # # Note that this is expensive (~20% slowdown if computed every 500 steps)
                # if step % 10000 == 0:
                #     sim = similarity.eval()
                #     for i in range(self.valid_size):
                #         valid_word = dataset.reversed_dictionary[self.valid_examples[i]]
                #         top_k = 8  # number of nearest neighbors
                #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #         log_str = 'Nearest to %s:' % valid_word
                #         for k in range(top_k):
                #             close_word = dataset.reversed_dictionary[nearest[k]]
                #             log_str = '%s %s,' % (log_str, close_word)
                #         print(log_str)

model = NERModel()
model.run_graph()