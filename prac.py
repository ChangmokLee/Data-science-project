# Import libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

from tensorflow.keras.callbacks import ModelCheckpoint
from google.colab import drive
import shutil

# Load file data
path_to_file = tf.keras.utils.get_file('austen.txt', 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/austen/austen.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))

# Verify the first part of our data
print(text[:200])

# Now we'll get a list of the unique characters in the file. This will form the
# vocabulary of our network. There may be some characters we want to remove from this
# set as we refine the network.
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(vocab)

# Next, we'll encode encode these characters into numbers so we can use them
# with our neural network, then we'll create some mappings between the characters
# and their numeric representations
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# Here's a little helper function that we can use to turn a sequence of ids
# back into a string:
# turn them into a string:
def text_from_ids(ids):
  joinedTensor = tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
  return joinedTensor.numpy().decode("utf-8")

# Now we'll verify that they work, by getting the code for "A", and then looking
# that up in reverse
testids = ids_from_chars(["T", "r", "u", "t", "h"])
testids

chars_from_ids(testids)

testString = text_from_ids( testids )
testString

# First, create a stream of encoded integers from our text
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
all_ids

# Now, convert that into a tensorflow dataset
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# Finally, let's batch these sequences up into chunks for our training
seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# This function will generate our sequence pairs:
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Call the function for every sequence in our list to create a new dataset
# of input->target pairs
dataset = sequences.map(split_input_target)

# Verify our sequences
for input_example, target_example in  dataset.take(1):
    print("Input: ", text_from_ids(input_example))
    print("--------")
    print("Target: ", text_from_ids(target_example))

# Finally, we'll randomize the sequences so that we don't just memorize the books
# in the order they were written, then build a new streaming dataset from that.
# Using a streaming dataset allows us to pass the data to our network bit by bit,
# rather than keeping it all in memory. We'll set it to figure out how much data
# to prefetch in the background.

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset

# Create our custom model. Given a sequence of characters, this
# model's job is to predict what character should come next.
class AustenTextModel(tf.keras.Model):

  # This is our class constructor method, it will be executed when
  # we first create an instance of the class
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)

    # 1. An embedding layer that handles the encoding of our vocabulary into
    #    a vector of values suitable for a neural network
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # 2. A GRU layer that handles the "memory" aspects of our RNN. If you're
    #    wondering why we use GRU instead of LSTM, and whether LSTM is better,
    #    take a look at this article: https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm
    #    then consider trying out LSTM instead (or in addition to!)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.gru2 = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)

    # 3. An LSTM layer. You need to specify return_sequences and return_state parameters.
    self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    self.lstm2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)

    # 4. Our output layer that will give us a set of probabilities for each
    #    character in our vocabulary.
    self.dense = tf.keras.layers.Dense(vocab_size)

  # This function will be executed for each epoch of our training. Here
  # we will manually feed information from one layer of our network to the
  # next.
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs

    # 1. Feed the inputs into the embedding layer, and tell it if we are
    #    training or predicting
    x = self.embedding(x, training=training)

    gru_states = []
    gru_states2 = []
    lstm_states = []
    lstm_states2 = []

    # 2. If we don't have any state in memory yet, get the initial random state
    #    from our GRUI layer.
    if states is None:
      gru_states = self.gru.get_initial_state(x)
      gru_states2 = self.gru.get_initial_state(x)
      lstm_states = self.lstm.get_initial_state(x)
      lstm_states2 = self.lstm.get_initial_state(x)
    else:
      gru_states = states[0]
      gru_states2 = states[1]
      lstm_states = states[2]
      lstm_states2 = states[3]

    # 3. Now, feed the vectorized input along with the current state of memory
    #    into the gru layer.
    x, gru_states = self.gru(x, initial_state=gru_states, training=training)

    x, gru_states2 = self.gru2(x, initial_state=gru_states2, training=training)

    # 4. Feed the results into the LSTM layer
    x, lstm_statesA, lstm_statesB = self.lstm(x, initial_state=lstm_states, training=training)
    lstm_states = [lstm_statesA, lstm_statesB]

    x, lstm_statesA, lstm_statesB = self.lstm2(x, initial_state=lstm_states2, training=training)
    lstm_states2 = [lstm_statesA, lstm_statesB]

    # 5. Finally, pass the results on to the dense layer
    x = self.dense(x, training=training)

    states = [gru_states, gru_states2, lstm_states, lstm_states2]

    # 6. Return the results
    if return_state:
      return x, states
    else:
      return x