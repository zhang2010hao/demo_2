import tensorflow as tf

lstm_hidden_size = 20
batch_sie = 20
num_steps = 10

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

state = lstm.zero_state(batch_sie, tf.float32)
loss = 0.0


def fully_connected(output):
    tf.nn.softmax(output)


def calc_loss(y, y_):
    y - y_


for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
        lstm_output, state = lstm(current_input, state)
        final_output = fully_connected(lstm_output)
        loss += calc_loss(final_output, expected_output)
