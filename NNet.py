from __future__ import print_function


import tensorflow as tf
import numpy as np

from sklearn.metrics import precision_score, recall_score


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def simpleNN(train_x, test_x, train_y, test_y, learn_rate, epochs, batch):
    # Parameters
    learning_rate = learn_rate
    training_epochs = epochs
    batch_size = batch
    display_step = 20

    new_test_y = np.zeros((len(test_y), 2))
    for i in range(len(test_y)):
        if test_y[i]== 0:
            new_test_y[i][0] = 1
        else:
            new_test_y[i][1] = 1

    new_train_y = np.zeros((len(train_y), 2))
    for i in range(len(train_y)):
        if train_y[i] == 0:
            new_train_y[i][0] = 1
        else:
            new_train_y[i][1] = 1


    m, n = train_x.shape
    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    n_hidden_2 = 256  # 2nd layer number of features
    n_input = n  # MNIST data input (img shape: 28*28)
    n_classes = 2  # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(m/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_beg = i*batch_size
                batch_end = (i+1)*batch_size
                batch_x = train_x[batch_beg: batch_end]
                batch_y = new_train_y[batch_beg: batch_end]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        test_pred = tf.argmax(pred, 1)
        test_label = tf.argmax(y, 1)
        correct_prediction = tf.equal(test_pred, test_label)
        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_x, y: new_test_y}))

        val_accuracy, y_pred = sess.run([accuracy, test_pred], feed_dict={x: test_x, y: new_test_y})
        print ("validation accuracy:", val_accuracy)
        y_true = np.argmax(new_test_y, 1)
        per = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print("Precision", per)
        print("Recall", recall)
        return val_accuracy, per, recall


