import os
import tensorflow as tf
from model.data_utils import batches


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)


    def save_session(self, step):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model, global_step=step)


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard

        batch_size = self.config.batch_size
        step = 0
        best_score = 0
        for i, (words, labels) in enumerate(batches(train, batch_size, self.config.nepochs)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            if step % 10000 == 0:
                self.config.lr *= self.config.lr_decay  # decay learning rate

            if step % 1000 == 0:
                metrics = self.run_evaluate(dev)
                msg = " - ".join(["{} {:04.2f}".format(k, v)
                                  for k, v in metrics.items()])
                self.logger.info('{}th step, Dev metrics {}'.format(step, msg))
                if metrics["f1"] >= best_score:
                    best_score = metrics['f1']
                    self.logger.info("- new best score!")
                    self.save_session(step)

            step = step + 1

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
