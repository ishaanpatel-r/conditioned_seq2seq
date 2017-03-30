import tensorflow as tf
import numpy as np

from datasets.friends import data
from datasets.friends import data_utils

class seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
            batch_size,
            model_name= 'naive_seq2seq',
            ckpt_path= 'ckpt/naive/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        def __graph__():
            # you know what this means
            tf.reset_default_graph()
            #
            # placeholders
            xs_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='xs')
            ys_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='ys') # decoder targets
            dec_inputs_ = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                        name='dec_inputs')

            # embed encoder input
            embs = tf.get_variable('emb', [vocab_size, state_size])
            enc_inputs = tf.nn.embedding_lookup(embs, xs_)

            # embed decoder input
            dec_inputs = tf.nn.embedding_lookup(embs, dec_inputs_)

            # define basic lstm cell
            basic_cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
            # add dropout
            #   dropout's keep probability
            keep_prob_ = tf.placeholder(tf.float32)

            # define lstm cell for encoder
            def lstm_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMCell(state_size, reuse=tf.get_variable_scope().reuse),
                    output_keep_prob=keep_prob_)


            with tf.variable_scope('encoder') as scope:
                # stack cells
                encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                           state_is_tuple=True)
                # define encoder
                enc_op, enc_context = tf.nn.dynamic_rnn(cell=encoder_cell, dtype=tf.float32, 
                                                  inputs=enc_inputs)

            with tf.variable_scope('decoder') as scope:
                decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                           state_is_tuple=True)
                # define decoder 
                dec_op, _ = tf.nn.dynamic_rnn(cell=decoder_cell, dtype=tf.float32,
                                              initial_state= enc_context,
                                              inputs=dec_inputs)
                
            ###    
            # predictions
            V = tf.get_variable('V', shape=[state_size, vocab_size], 
                                initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable('bo', shape=[vocab_size], 
                                 initializer=tf.constant_initializer(0.))
            ####
            # flatten states to 2d matrix for matmult with V
            dec_op_reshaped = tf.reshape(dec_op, [-1, state_size])
            # /\_o^o_/\
            logits = tf.matmul(dec_op_reshaped, V) + bo
            #
            # predictions
            predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=tf.reshape(ys_, [-1]))
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
            #
            # attach symbols to class
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.keep_prob_ = keep_prob_
            self.xs_ = xs_
            self.ys_ = ys_
            self.dec_inputs_ = dec_inputs_
            # attach session to self
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            #####
        ####
        # build graph
        __graph__()


    def restore_last_checkpoint(self):
        saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            print('>> Restoring last checkpoint : ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)


    def predict(self, query):
        # query -> array of indices
        #
        # build feed dict
        feed_dict = {
                self.xs_ : query,
                self.keep_prob_ : 1.
                }
        return self._sess.run(self.predictions, feed_dict = feed_dict)


    def train(self, trainset, testset, n, epochs, valid_n, eval_interval=10, save=False):

        print('\n>> Training begins!\n')

        def fetch_dict(datagen, keep_prob=0.5):
            bx, by, _ = datagen.__next__()
            by_dec = np.zeros_like(by).T
            by_dec[1:] = by.T[:-1]

            feed_dict = {
                self.xs_: bx,
                self.ys_: by,
                self.dec_inputs_ : by_dec.T,
                self.keep_prob_: keep_prob
            }
            return feed_dict

        ##
        # setup session
        saver = tf.train.Saver()
        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            print('>> Restoring last checkpoint : ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)

        try:
            # start training
            for j in range(epochs):
                mean_loss = 0
                for i in range(n):
                    _, l = self._sess.run([self.train_op, self.loss], 
                            feed_dict = fetch_dict(trainset) 
                            )
                    mean_loss += l
                    sys.stdout.write('[{}/{}]\r'.format(i,n))

                print('>> [{j}] train loss at : {}'.format(j, mean_loss/n))
                if j and j%eval_interval == 0:
                    if save:
                        saver.save(self._sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    #
                    # evaluate
                    validloss = 0
                    for k in range(valid_n):
                        validloss = self._sess.run(self.loss, 
                                feed_dict = fetch_dict(validset, keep_prob=1.)
                                )
                    print('valid loss : {}'.format(validloss / valid_n))
 
        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))
