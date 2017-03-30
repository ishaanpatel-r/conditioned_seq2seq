import tensorflow as tf

import sys


class conditioned_seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
                 ext_context_size,
                 batch_size,
                 model_name='conditioned_seq2seq',
                 ckpt_path='ckpt/conditioned_seq2seq/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        def __graph__():
            # you know what this means
            tf.reset_default_graph()

            # placeholders
            xs_ = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                 name='xs')
            ys_ = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                 name='ys')  # decoder targets
            dec_inputs_length_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                                                name='dec_inputs_length')
            ext_context_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                                          name='ext_context')

            # embed encoder input
            embs = tf.get_variable('emb', [vocab_size, state_size])
            enc_inputs = tf.nn.embedding_lookup(embs, xs_)

            # define lstm cell for encoder
            encoder_cell = tf.contrib.rnn.LSTMCell(state_size)
            # add dropout
            #   dropout's keep probability
            keep_prob_ = tf.placeholder(tf.float32)

            # define lstm cell for encoder
            def lstm_cell():
                return tf.contrib.rnn.DropoutWrapper(
                   # for regular version
                    tf.contrib.rnn.LSTMCell(state_size)
                   # for bleeding edge tf version
                   # tf.contrib.rnn.LSTMCell(state_size, reuse=tf.get_variable_scope().reuse),
                   # output_keep_prob=keep_prob_
                )

            # stack cells
            encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                       state_is_tuple=True)

            # define encoder
            enc_op, enc_context = tf.nn.dynamic_rnn(cell=encoder_cell, dtype=tf.float32, inputs=enc_inputs)

            # R value classifier
            Vr = tf.get_variable('Vr', shape=[num_layers*state_size, ext_context_size], 
                                initializer=tf.contrib.layers.xavier_initializer())
            br = tf.get_variable('br', shape=[ext_context_size], 
                                 initializer=tf.constant_initializer(0.))
            def r_classifier():
                enc_context_c = tf.concat([c for c,h in enc_context], axis=1)
                r_logits = tf.matmul(enc_context_c, Vr) + br
                self.r_preds = tf.argmax(tf.nn.softmax(r_logits), axis=1) # attach to self
                r_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=r_logits, labels=ext_context_)
                self.r_loss = tf.reduce_mean(r_losses)
                self.train_op_r = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.r_loss)

            # add r_classifier to graph
            r_classifier()

            # embed ext_context
            ext_ctxt_embs = tf.get_variable('ext_ctxt_emb', [ext_context_size, state_size])
            # get a slice from embedding matrix
            ext_context = tf.nn.embedding_lookup(ext_ctxt_embs, self.r_preds)


            # output projection
            V = tf.get_variable('V', shape=[state_size, vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable('bo', shape=[vocab_size],
                                 initializer=tf.constant_initializer(0.))

            ###
            # embedding for pad symbol
            PAD = tf.nn.embedding_lookup(embs, tf.zeros(shape=[tf.shape(xs_)[0], ],
                                                        dtype=tf.int32))

            ###
            # init function for raw_rnn
            def loop_fn_initial(time, cell_output, cell_state, loop_state):
                assert cell_output is None and loop_state is None and cell_state is None

                elements_finished = (time >= dec_inputs_length_)
                initial_input = PAD
                initial_cell_state = enc_context
                initial_loop_state = None

                return (elements_finished,
                        initial_input,
                        initial_cell_state,
                        None,
                        initial_loop_state)

            ###
            # state transition function for raw_rnn
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_state is None:  # time == 0
                    return loop_fn_initial(time, cell_output, cell_state, loop_state)

                emit_output = cell_output  # == None for time == 0
                #
                # couple external context with cell states (c, h)
                next_cell_state = []
                for layer in range(num_layers):
                    next_cell_state.append(tf.contrib.rnn.LSTMStateTuple(
                        c=cell_state[layer].c + ext_context,
                        h=cell_state[layer].h + ext_context
                    ))

                next_cell_state = tuple(next_cell_state)

                elements_finished = (time >= dec_inputs_length_)
                finished = tf.reduce_all(elements_finished)

                def search_for_next_input():
                    output = tf.matmul(cell_output, V) + bo
                    prediction = tf.argmax(output, axis=1)
                    return tf.nn.embedding_lookup(embs, prediction)

                next_input = tf.cond(finished, lambda: PAD, search_for_next_input)

                next_loop_state = None

                return (elements_finished,
                        next_input,
                        next_cell_state,
                        emit_output,
                        next_loop_state)

            with tf.variable_scope('decoder') as dec_scope:
                ###
                # define the decoder with raw_rnn <- loop_fn, loop_fn_initial
                decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                           state_is_tuple=True)
                decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()

            ####
            # flatten states to 2d matrix for matmult with V
            dec_op_reshaped = tf.reshape(decoder_outputs, [-1, state_size])
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
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            #
            # attach symbols to class
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.ext_context_size = ext_context_size
            self.keep_prob_ = keep_prob_  # placeholders
            self.xs_ = xs_
            self.ys_ = ys_
            self.dec_inputs_length_ = dec_inputs_length_
            self.ext_context_ = ext_context_
            # and finally, attach session to self
            init_op = tf.global_variables_initializer()
            self._sess = tf.Session()
            self._sess.run(init_op)


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
                self.dec_inputs_length_ : [query.shape[-1]*2], # this bothers me!
                self.keep_prob_ : 1.
                }
        return self._sess.run(self.predictions, feed_dict = feed_dict)


    def train(self, trainset, validset, n, epochs, valid_n, eval_interval=10, save=False):

        print('\n>> Training begins!\n')

        def fetch_dict(datagen, keep_prob=0.5):
            qi, ai, ri = datagen.__next__()
            # non-zero elements in each example of ai
            dec_lengths = (ai != 0).sum(1)

            feed_dict = {
                self.xs_: qi,
                self.ys_: ai,
                self.dec_inputs_length_: dec_lengths,
                self.ext_context_: ri,
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
                    _, _, l = self._sess.run([self.train_op, self.train_op_r, self.loss],
                                    feed_dict=fetch_dict(trainset)
                                    )
                    mean_loss += l
                    sys.stdout.write('[{}/{}]\r'.format(i, n))
                print('\n>> [{}] train loss at : {}'.format(j, mean_loss / n))

                if j and j%eval_interval == 0:
                    if save:
                        saver.save(self._sess, self.ckpt_path + self.model_name + '.ckpt', global_step=j)
                    #
                    # evaluate
                    validloss = 0
                    for k in range(valid_n):
                        validloss += self._sess.run(self.loss,
                                             feed_dict=fetch_dict(validset, keep_prob=1.)
                                             )
                    print('valid loss : {}'.format(validloss / valid_n))

        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))
