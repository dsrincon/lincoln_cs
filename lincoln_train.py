
# coding: utf-8

# # Lincoln: Stock performance predictor based on news
# Lincon is an intelligent AI system that helps professional and retail investors get ahead of the market by analizing news from different sources around the Internet and predicting the behavoir of specific stocks by interpreting the the content of these news sources. This project trains the Lincoln AI using 1 layer LSTM Recurrent Neural netwokr architecture using daily stock data for 300 NASDAQ companies and news from: cnn.com, wsj.com, forbes.com, marketwatch.com, thestreet.com, thisismoney.co.uk, kiplinger.com, bloomberg.com, highpointobserver.com
# 

# In[ ]:

import numpy as np
import tensorflow as tf
import codecs


# In[ ]:

Load historical news and stock price data. 


# In[ ]:

with codecs.open('news.txt', 'r', encoding='utf-8') as f:
    reviews = f.read()
with codecs.open('results.txt', 'r', encoding='utf-8') as f:
    labels = f.read()


# In[ ]:

#Example news extract
reviews[:2000]


# ## Data preprocessing
# 
# We preprocess the data to get rid of punctuation marks to simplify vocabulary

# In[ ]:

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
#all_text=''.join([i for i in all_text if not i.isnumeric()])
reviews = all_text.split('\r')

all_text = ' '.join(reviews)
words = all_text.split()


# In[ ]:

#vocabulary length
words[10000:10100]
print(len(words))


# ### Encoding the words
# 
# The embedding lookup requires that we pass in integers to our network. The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews into integers so they can be passed into the network.

# In[ ]:

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])


# ### Encoding the labels
# 
# Our labels are the results file has values of  "UP" or "DOWN" depending on how the stock price closed on a given day. To use these labels in our network, we need to convert them to 0 and 1.

# In[ ]:

labels = labels.split('\r')
labels = np.array([1 if each == "UP" else 0 for each in labels])


# In[ ]:

#Review length sizes
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Now we create an array `features` that contains the data we'll pass to the network. The data should come from `review_ints`, since we want to feed integers to the network. Each row should be 1.500 elements long. For reviews shorter than 1500 words, left pad with 0s. That is, if the review is `['best', 'movie', 'ever']`, `[117, 18, 128]` as integers, the row will look like `[0, 0, 0, ..., 0, 117, 18, 128]`. For reviews longer than 1500, use on the first 1500 words as the feature vector.
# 
# 
# 

# In[ ]:

seq_len = 1500
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]


# ## Training, Validation, Test
# 
# 

# With our data in nice shape, we'll split it into training (80%), validation(10%), and test sets (10%).
# 
# 

# In[ ]:

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# ## Build the graph
# 
# Here, we'll build the graph. First up, defining the hyperparameters.
# 
# * `lstm_size`: Number of units in the hidden layers in the LSTM cells. 
# * `lstm_layers`: Number of LSTM layers in the network. 
# * `batch_size`: The number of reviews to feed the network in one training pass.
# * `learning_rate`: Learning rate

# In[ ]:

lstm_size = 256
lstm_layers = 1
batch_size = 100
learning_rate = 0.001
tf.reset_default_graph()


# For the network itself, we'll be passing in our 1500 element long review vectors. Each batch will be `batch_size` vectors. We'll also be using dropout on the LSTM layer, so we'll make a placeholder for the keep probability.

# In[ ]:

n_words = len(vocab_to_int)

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# ### Embedding
# 
# Now we'll add an embedding layer. We need to do this because there are 74000 words in our vocabulary. It is massively inefficient to one-hot encode our classes.  Instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. We'll add a new layer and let the network learn the weights.
# 
# 
# 
# 

# In[ ]:

# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# ### LSTM cell
# 
# We're using a single LSTM cell with dropout to avoid overfitting. We're also initializing the LSTM state. 
# 

# In[ ]:

with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# ### RNN forward pass
# 
# 

# In[ ]:

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)


# ### Output
# 
# We only care about the final output of our LSTM, we'll be using that as our price prediction. So we need to grab the last output with `outputs[:, -1]`, the calculate the cost from that and `labels_`.

# In[ ]:

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ### Validation accuracy
# 
# Here we can add a few nodes to calculate the accuracy which we'll use in the validation pass.

# In[ ]:

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ### Batching
# 
# This is a simple function for returning batches from our data. First it removes data such that we only have full batches. Then it iterates through the `x` and `y` arrays and returns slices out of those arrays with size `[batch_size]`.

# In[ ]:

def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# ## Training
# 
#     We'll be running 10 training epochs to achive the lowest possible validation loss. 

# In[ ]:

epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
            saver.save(sess, "modelo")
    saver.save(sess, "checkpoints/sentiment.ckpt")
    


# ## Testing

# In[ ]:

test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

