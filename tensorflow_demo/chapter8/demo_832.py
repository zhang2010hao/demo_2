import tensorflow as tf

lstm_size = 20
layer_num = 5
batch_size = 30
num_steps = 40

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size)) for _ in range(layer_num)])