import tensorflow as tf

class MyNN:
    def __init__(self, action_size, state_size, learning_rate):
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = learning_rate

        self.input = tf.placeholder(tf.float32, [None, *state_size])
        self.actions = tf.placeholder(tf.float32, [None, action_size])
        self.target_Q = tf.placeholder(tf.float32)

        self.conv1 = tf.layers.conv2d(
            inputs = self.input,
            filters = 32,
            kernel_size = [8,8],
            strides = [4,4],
            padding = "VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name = "conv1")

        self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

        self.conv2 = tf.layers.conv2d(
            inputs = self.conv1_out,
            filters = 64,
            kernel_size = [4,4],
            strides = [2,2],
            padding = "VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name = "conv2")

        self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

        self.conv3 = tf.layers.conv2d(
            inputs = self.conv2_out,
            filters = 64,
            kernel_size = [3,3],
            strides = [2,2],
            padding = "VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name = "conv3")

        self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

        self.flatten = tf.contrib.layers.flatten(self.conv3_out)

        self.fc = tf.layers.dense(
            inputs = self.flatten,
            units = 512,
            activation = tf.nn.elu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="fc1")

        self.output = tf.layers.dense(
            inputs = self.fc,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            units = self.action_size,
            activation=None)

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)