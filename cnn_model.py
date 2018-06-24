import tensorflow as tf


class CNN_Categorizer_v2:
    def __init__(self, sentence_length, num_classes, vocabulary_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0, embedding_type="static", pretrained_embedding=None):
        self.sentence_length = sentence_length
        self.num_classes = num_classes
        self.vocabulary_size = vocabulary_size
        self.pretrained_embedding = pretrained_embedding
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_type = embedding_type

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_norm = tf.placeholder(tf.bool, name="batch_norm")

        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocabulary_size, embedding_size],
                                                    name="pretrained_embedding")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Training embedding stuff
            if self.embedding_type == "static":
                embedding_ = tf.Variable(tf.constant(0.0, shape=[self.vocabulary_size, embedding_size]),
                                         trainable=False,
                                         name='embedding_weight')
                embedding_train = tf.assign(embedding_, self.embedding_placeholder)
            elif self.embedding_type == "nonstatic":
                embedding_ = tf.Variable(tf.constant(0.0, shape=[self.vocabulary_size, embedding_size]),
                                         trainable=True,
                                         name='embedding_weight')
                embedding_train = tf.assign(embedding_, self.embedding_placeholder)
            elif self.embedding_type == "random":
                embedding_train = tf.Variable(tf.truncated_normal([self.vocabulary_size+1, self.embedding_size],
                                                            name='pretrained_embedding'))
            elif self.embedding_type == "multichannel":
                raise NotImplementedError("No multichannel implementation")
            else:
                print("Invalid empedding type:", self.embedding_type)

            # Validation embedding must be non-trainable!
            embedding_ = tf.Variable(tf.constant(0.0, shape=[self.vocabulary_size, embedding_size]),
                                     trainable=False,
                                     name='embedding_weight_vali')
            embedding_validation = tf.assign(embedding_, self.embedding_placeholder)

            embedding = tf.cond(self.batch_norm, lambda: embedding_train, lambda: embedding_validation)

            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)
            self.static_embedding_input_expanded = tf.expand_dims(self.embedding_input, -1)

        with tf.device('/gpu:0'):
            layers = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-" + str(filter_size)):
                    conv_filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    pool_filter_shape = [1, self.sentence_length - filter_size + 1, 1, 1]
                    weight = self._get_weights(conv_filter_shape)
                    bias = self._get_bias([self.num_filters])
                    conv = self._set_convolution(self.static_embedding_input_expanded, weight) + bias
                    conv_bn = tf.layers.batch_normalization(conv, training=self.batch_norm)
                    h = tf.nn.relu(conv_bn)
                    pool = self._set_pool(h, pool_filter_shape)
                    layers.append(pool)

            total_num_filters = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(layers, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])

            self.h_dropout = tf.nn.dropout(self.h_pool_flat, keep_prob=self.keep_prob)

            with tf.name_scope("fully_connected_layer1"):
                weight_fc1 = self._get_weights(shape=[total_num_filters, 128])
                bias_fc1 = self._get_bias(shape=[128])
                fc1 = tf.nn.xw_plus_b(self.h_dropout, weight_fc1, bias_fc1)
                fc1_bn = tf.layers.batch_normalization(fc1, training=self.batch_norm)
                fc1_out = tf.nn.tanh(fc1_bn, name="fc1_out")
                fc1_dropout = tf.nn.dropout(fc1_out, keep_prob=self.keep_prob)
            #
            # with tf.name_scope("fully_connected_layer2"):
            #     weight_fc2 = self._get_weights(shape=[128, 64])
            #     bias_fc2 = self._get_bias(shape=[64])
            #     fc2 = tf.nn.xw_plus_b(fc1_dropout, weight_fc2, bias_fc2)
            #     fc2_bn = tf.layers.batch_normalization(fc2, training=self.batch_norm)
            #     fc2_out = tf.nn.relu(fc2_bn, name="fc2_out")
            #     fc2_drop = tf.nn.dropout(fc2_out, self.keep_prob)

            with tf.name_scope("output"):
                weight_out = self._get_weights(shape=[128, self.num_classes])
                bias_out = self._get_bias(shape=[num_classes])
                output = tf.nn.xw_plus_b(fc1_dropout, weights=weight_out, biases=bias_out, name="logits")

            with tf.name_scope("loss"):
                # losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=output)
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=output)
                l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                     name="l2_losses") * l2_reg_lambda
                losses = tf.reduce_mean(losses, name="softmax_loss")
                self.loss = tf.add(losses, l2_losses, name="loss")

            # Accuracy
            with tf.name_scope("accuracy"):
                self.predictions = tf.argmax(output, 1, name="predictions")
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

            with tf.name_scope("accuracytopk"):
                pred_softmax = tf.nn.softmax(output)
                pred_sigmoid = tf.sigmoid(output)
                self.pred_top_2 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(pred_sigmoid, tf.argmax(self.input_y, 1), k=2), tf.float32), name="top2")
                self.pred_top_3 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(pred_sigmoid, tf.argmax(self.input_y, 1), k=3), tf.float32), name="top3")
                self.pred_top_4 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(pred_sigmoid, tf.argmax(self.input_y, 1), k=4), tf.float32), name="top4")
                self.pred_top_5 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(pred_sigmoid, tf.argmax(self.input_y, 1), k=5), tf.float32), name="top5")

    def _set_convolution(self, input, weight, strides=[1, 1, 1, 1], padding="VALID"):
        return tf.nn.conv2d(input, weight, strides=strides, padding=padding, name="conv")

    def _set_pool(self, input, shape, strides=[1, 1, 1, 1], padding='VALID'):
        return tf.nn.max_pool(input, ksize=shape, strides=strides, padding=padding, name="pool")

    def _get_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="w")

    def _get_bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape), name="b")
