import tensorflow as tf

y = ['a', 'b', 'c', 'd', 'e']
digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
pack = zip(y, digits)
for y, digit in pack:
    print(y, digit.numpy())

