import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load Iris dataset for Problem 1 and Problem 3
df = pd.read_csv("Iris.csv")

# Problem 1: Binary Classification (Iris Versicolor vs Iris Virginica)
# Filter dataset to use only Iris-versicolor and Iris-virginica
df_binary = df[(df["Species"] == "Iris-versicolor") | (df["Species"] == "Iris-virginica")]
y_binary = df_binary["Species"]
X_binary = df_binary.loc[:, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Convert categorical labels to numeric
y_binary[y_binary == "Iris-versicolor"] = 0
y_binary[y_binary == "Iris-virginica"] = 1
y_binary = y_binary.astype(np.int64)[:, np.newaxis]

# Train-test split for binary classification
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Problem 2: TensorFlow - 2-class Classification
# Define the batch generator class
class GetMiniBatch:
    def __init__(self, X, y, batch_size=10, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0] / self.batch_size).astype(np.int)

    def __len__(self):
        return self._stop

    def __getitem__(self, item):
        p0 = item * self.batch_size
        p1 = item * self.batch_size + self.batch_size
        return self.X[p0:p1], self.y[p0:p1]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter * self.batch_size
        p1 = self._counter * self.batch_size + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]

# Hyperparameters for the binary classification task
learning_rate = 0.001
batch_size = 10
num_epochs = 100
n_hidden1 = 50
n_hidden2 = 100
n_input = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = 1  # Binary classification

# Define the placeholders for TensorFlow
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Batch generator for training
get_mini_batch_train = GetMiniBatch(X_train, y_train, batch_size=batch_size)

def example_net(x):
    tf.random.set_random_seed(0)
    weights = {
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'w3': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden1])),
        'b2': tf.Variable(tf.random_normal([n_hidden2])),
        'b3': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return layer_output

# Neural network and training setup
logits = example_net(X)
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.sign(Y - 0.5), tf.sign(tf.sigmoid(logits) - 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialization and training
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_batch = np.ceil(X_train.shape[0] / batch_size).astype(np.int64)
        total_loss = 0
        total_acc = 0
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):
            sess.run(train_op, feed_dict={X: mini_batch_x, Y: mini_batch_y})
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: mini_batch_x, Y: mini_batch_y})
            total_loss += loss
        total_loss /= n_samples
        val_loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_val, Y: y_val})
        print(f"Epoch {epoch}, loss : {total_loss:.4f}, val_loss : {val_loss:.4f}, acc : {acc:.3f}")
    test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})
    print(f"test_acc : {test_acc:.3f}")

# Problem 3: Multi-class Classification (All three classes of Iris)
df_multi = df
y_multi = df_multi["Species"]
X_multi = df_multi.loc[:, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Convert categorical labels to numeric for multi-class
y_multi = pd.get_dummies(y_multi).values  # One-hot encoding for 3 classes
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Redefine network and loss function for multi-class classification
def example_net_multi_class(x):
    tf.random.set_random_seed(0)
    weights = {
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'w3': tf.Variable(tf.random_normal([n_hidden2, 3]))  # 3 classes
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden1])),
        'b2': tf.Variable(tf.random_normal([n_hidden2])),
        'b3': tf.Variable(tf.random_normal([3]))  # 3 classes
    }

    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return layer_output

# Define softmax cross-entropy for multi-class classification
logits_multi = example_net_multi_class(X)
loss_op_multi = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits_multi))
optimizer_multi = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_multi = optimizer_multi.minimize(loss_op_multi)
correct_pred_multi = tf.equal(tf.argmax(Y, 1), tf.argmax(tf.nn.softmax(logits_multi), 1))
accuracy_multi = tf.reduce_mean(tf.cast(correct_pred_multi, tf.float32))

# Initialization and training for multi-class classification
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_batch = np.ceil(X_train.shape[0] / batch_size).astype(np.int64)
        total_loss = 0
        total_acc = 0
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):
            sess.run(train_op_multi, feed_dict={X: mini_batch_x, Y: mini_batch_y})
            loss, acc = sess.run([loss_op_multi, accuracy_multi], feed_dict={X: mini_batch_x, Y: mini_batch_y})
            total_loss += loss
        total_loss /= n_samples
        val_loss, acc = sess.run([loss_op_multi, accuracy_multi], feed_dict={X: X_val, Y: y_val})
        print(f"Epoch {epoch}, loss : {total_loss:.4f}, val_loss : {val_loss:.4f}, acc : {acc:.3f}")
    test_acc_multi = sess.run(accuracy_multi, feed_dict={X: X_test, Y: y_test})
    print(f"test_acc (multi-class): {test_acc_multi:.3f}")

# Problem 4: House Prices Regression (predict SalePrice)
df_house = pd.read_csv("train.csv")  # Assuming this file exists
X_house = df_house[['GrLivArea', 'YearBuilt']]  # Using selected features
y_house = df_house['SalePrice']

X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(X_house, y_house, test_size=0.2, random_state=0)

# Define a simple regression model for house prices
X_house_input = tf.placeholder("float", [None, X_train_house.shape[1]])
y_house_input = tf.placeholder("float", [None, 1])

# Define a linear model for regression
weights_house = tf.Variable(tf.random_normal([X_train_house.shape[1], 1]))
biases_house = tf.Variable(tf.random_normal([1]))
pred_house = tf.add(tf.matmul(X_house_input, weights_house), biases_house)

# Loss and optimizer for regression
loss_op_house = tf.reduce_mean(tf.square(pred_house - y_house_input))
optimizer_house = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_house = optimizer_house.minimize(loss_op_house)

# Training loop for house price prediction
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_loss_house = 0
        for i in range(len(X_train_house) // batch_size):
            mini_batch_x_house = X_train_house[i * batch_size: (i + 1) * batch_size]
            mini_batch_y_house = y_train_house[i * batch_size: (i + 1) * batch_size]
            sess.run(train_op_house, feed_dict={X_house_input: mini_batch_x_house, y_house_input: mini_batch_y_house})
            loss = sess.run(loss_op_house, feed_dict={X_house_input: mini_batch_x_house, y_house_input: mini_batch_y_house})
            total_loss_house += loss
        print(f"Epoch {epoch}, loss : {total_loss_house / len(X_train_house)}")

    test_loss_house = sess.run(loss_op_house, feed_dict={X_house_input: X_test_house, y_house_input: y_test_house})
    print(f"test_loss (House Price): {test_loss_house}")

# Problem 5: MNIST Classification
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# MNIST data shape
n_input_mnist = 784  # 28x28 pixels
n_classes_mnist = 10  # 10 digits (0-9)

# Define the placeholders for MNIST input and output
X_mnist = tf.placeholder("float", [None, n_input_mnist])
Y_mnist = tf.placeholder("float", [None, n_classes_mnist])

# Define the MNIST neural network structure
weights_mnist = {
    'w1': tf.Variable(tf.random_normal([n_input_mnist, 256])),
    'w2': tf.Variable(tf.random_normal([256, 128])),
    'w3': tf.Variable(tf.random_normal([128, n_classes_mnist]))
}
biases_mnist = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([128])),
    'b3': tf.Variable(tf.random_normal([n_classes_mnist]))
}

# Neural network layers for MNIST
def mnist_net(x):
    layer_1 = tf.add(tf.matmul(x, weights_mnist['w1']), biases_mnist['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights_mnist['w2']), biases_mnist['b2'])
    layer_2 = tf.nn.relu(layer_2)
    output_layer = tf.matmul(layer_2, weights_mnist['w3']) + biases_mnist['b3']
    return output_layer

# Loss and optimizer for MNIST classification
logits_mnist = mnist_net(X_mnist)
loss_op_mnist = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_mnist, logits=logits_mnist))
optimizer_mnist = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_mnist = optimizer_mnist.minimize(loss_op_mnist)

# Initialization and training for MNIST classification
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_batch_mnist = np.ceil(mnist.train.num_examples / batch_size).astype(np.int64)
        for i in range(total_batch_mnist):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op_mnist, feed_dict={X_mnist: batch_x, Y_mnist: batch_y})

        print(f"Epoch {epoch}, loss : {sess.run(loss_op_mnist, feed_dict={X_mnist: batch_x, Y_mnist: batch_y})}")

    test_acc_mnist = sess.run(accuracy_multi, feed_dict={X_mnist: mnist.test.images, Y_mnist: mnist.test.labels})
    print(f"test_acc (MNIST): {test_acc_mnist:.3f}")
