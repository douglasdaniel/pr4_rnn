#!/usr/bin/env python3

import pickle, pdb, random
import numpy as np
import tensorflow as tf
from aux import accuracy
from model import model



DISPLAY_STEP = 100
MAX_ITERATIONS = 300
rand = random.sample(range(MAX_ITERATIONS), MAX_ITERATIONS)
N_TRAIN_SEQ = 7500
N_TEST_SEQ = 500
batch_size = 3
LEARN_RATE = 0.2



# load the dataset into memory
with open('./data/youtube_train_data.pkl','rb') as data_file:
    data,labels = pickle.load(data_file)

data = data.astype(np.float32)
labels = labels.astype(np.float32)

# normalize the training data over each image
data -= np.mean(data, axis=(2,3,4), keepdims=True)
data /= np.std(data, axis=(2,3,4), keepdims=True)

# Split dataset into training and testing
train_data = data[:N_TRAIN_SEQ]
train_labels = labels[:N_TRAIN_SEQ]
test_data = data[-N_TEST_SEQ:]
test_labels = labels[-N_TEST_SEQ:]


# Instantiate the graph
graph = tf.Graph()
with graph.as_default():

    # Input data
    tf_train_data = tf.placeholder(shape=[batch_size, 10, 64, 64, 3], dtype=tf.float32,name='traindata')
    tf_train_labels = tf.placeholder(shape=[batch_size, 10, 7, 2], dtype=tf.float32,name='trainlabels')
    tf_test_data = tf.constant(test_data,dtype=tf.float32)

    # Compute predictions using the model
    predict_op= model(tf_train_data)
    test_op= model(tf_test_data)

    # Compute the loss
    loss = tf.losses.mean_squared_error(labels=tf_train_labels,predictions=predict_op)
    tf.summary.scalar("loss",loss)

    # Instantiate optimizer to calculate gradients
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    # Save the model
    tf.get_collection('validation_nodes')

    # Add opts
    tf.add_to_collection('validation_nodes', tf_train_data)
    tf.add_to_collection('validation_nodes', tf_train_labels)
    tf.add_to_collection('validation_nodes', predict_op)

    # Begin training
    saver = tf.train.Saver()

    merge = tf.summary.merge_all()
with tf.Session(graph=graph) as session:
    session.run(init_op)
    
    writer = tf.summary.FileWriter('./output', session.graph)
    print('Initialized')
    
    for step in np.arange(MAX_ITERATIONS+1):  
        # Get new batch of data
        offset = (rand[step] * batch_size) % (train_labels.shape[0] - batch_size)
        data_batch = train_data[offset:(offset+batch_size),:]
        label_batch = train_labels[offset:(offset+batch_size),:]

        # Update input data
        feed_dict = {tf_train_data : data_batch, tf_train_labels : label_batch}
         
        # Run training
        summary,predictions,lass,_ = session.run([merge, predict_op, loss, optimizer],feed_dict=feed_dict)
        writer.add_summary(summary, step)
        
        # Compute average pixel distance error
        summary2 = tf.Summary()
        train_apde,_ = accuracy(predictions, label_batch)
        summary2.value.add('train apde', simple_value=train_apde)
        
        if step % DISPLAY_STEP == 0:
            # Compute the average pixed distance error on both sets
            # Compute the average error for each joint on test set
            test_predictions = session.run(test_op)
            test_apde,errPerJnt = accuracy(test_predictions,test_labels)
            summary2.value.add('test apde', simple_value=test_apde)
        
            message = "step {:4d} : loss is {:6.2f}, training APDE= {:2.2f} px, testing APDE= {:2.2f} px".format(step, lass, train_apde, test_apde)
            print(message)


        writer.add_summary(summary2, step)
    
        save_path = saver.save(session, "./output/my_model")
    # Exit Gracefully
