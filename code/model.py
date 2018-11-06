from collections import Counter
import tensorflow as tf
import numpy as np
import logging
from general_utils import print_sentence
from data_utils import minibatches, pad_sequence, get_chunks
import sys, time

class Model(object):
    def __init__(self, config, ntags, n_words, logger=None):
        self.config = config
        # self.embeddings = None
        self.ntags = ntags

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', label=logging.DEBUG)
        
        self.logger = logger
        self.n_words = n_words


    def add_placeholder(self):
        self.sentence_ids = tf.placeholder(tf.int32, shape=[None, None], name="sentence_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')


    def get_feed_dict(self, sentences, labels=None, learning_rate=None, dropout=None):
        sentences_ids, sequence_lengths = pad_sequence(sentences, 0)
        feed = {
            self.sentence_ids:sentences_ids,
            self.sequence_lengths:sequence_lengths
        }
        if labels is not None:
            labels, _ = pad_sequence(labels, 0)
            feed[self.labels] = labels
        
        if dropout is not None:
            feed[self.dropout] = dropout
        
        if learning_rate is not None:
            feed[self.learning_rate] = learning_rate
        
        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(tf.truncated_normal(shape=[self.n_words, self.config.n_dim], dtype=tf.float32),
                                           dtype=tf.float32,
                                           trainable=self.config.train_embeddings,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.sentence_ids, name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    
    def add_biLSTM_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw, 
                inputs = self.word_embeddings, 
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", 
                                shape=[2 * self.config.hidden_size, self.ntags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="b", 
                                shape=[self.ntags], 
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

    
    def add_pred_op(self):
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
    

    def add_loss_op(self):
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.labels,
                sequence_lengths=self.sequence_lengths
            )
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses= tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)
        
    
    def add_train_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
    

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    
    def build(self):
        self.add_placeholder()
        self.add_word_embeddings_op()
        self.add_biLSTM_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()


    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.summary_path, sess.graph)
    

    def train(self, train, dev, vocab_tags, vocab_words):
        
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(self.init)
            self.add_summary(sess)

            for epoch in range(self.config.n_epoch):
                # self.logger.info("Epoch {:} out of {:}".format(epoch+1, self.config.n_epoch))
                acc = self.run_epoch(sess, train, dev, vocab_tags, vocab_words, epoch, saver)

                self.config.learning_rate *= self.config.learning_rate_decay



    def predict_batch(self, sess, sentences):
        fd, sequence_lengths = self.get_feed_dict(sentences, dropout=1.0)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                feed_dict=fd)

            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit[:sequence_length], transition_params
                )
                viterbi_sequences += [viterbi_sequence]
            
            return viterbi_sequences, sequence_lengths
        
        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths



    def run_epoch(self, sess, train, dev, vocab_tags, vocab_words, epoch, saver):
        """
        performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            vocab_tags: {tag: index} dictionary
            epoch:(int)
            saver: tf saver instance
        """
        num_batches = (len(train)+self.config.batch_size -1) // self.config.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        for step, (sentences, labels) in enumerate(minibatches(train, vocab_tags, vocab_words, self.config.batch_size)):
            sys.stdout.write(" processing: {} batch / {} batches.".format(step+1, num_batches)+"\r")
            step_num = self.config.n_epoch * num_batches + step + 1
            
            fd, _ = self.get_feed_dict(sentences, labels, self.config.learning_rate, self.config.dropout)
            
            _, train_loss, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                            feed_dict=fd)

            if step + 1==1 or (step+1) % 100 == 0 or step+1 == num_batches:
                self.logger.info("{} epoch {}, step {}, loss: {:.4}, f1: {:.4}, global_step:{}".format(
                    start_time, epoch+1, step+1, train_loss, self.run_evaluate(sess, dev, vocab_tags, vocab_words)["f1"], step_num_
                ))

            self.file_writer.add_summary(summary, step_num_)

            if step+1 == num_batches:
                saver.save(sess, self.config.model_path, global_step=step_num_)

        metrics = self.run_evaluate(sess, dev, vocab_tags, vocab_words)       
        msg = " - ".join(["{} {:04.4f}".format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        
        return metrics["f1"]

    
    def run_evaluate(self, sess, test, vocab_tags, vocab_words):

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for sentences, labels in minibatches(test, vocab_tags, vocab_words, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, sentences)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred= lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p *r /(p+r) if correct_preds > 0 else 0
        acc = np.mean(accs)
              
        return {"acc":100*acc, "f1":100*f1}

    
    
    



        