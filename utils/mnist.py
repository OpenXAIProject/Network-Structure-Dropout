from tensorflow.examples.tutorials.mnist import input_data
from paths import MNIST_PATH

def mnist_input(batch_size):
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    n_train_batches = mnist.train.num_examples/batch_size
    n_test_batches = mnist.test.num_examples/batch_size
    return mnist, n_train_batches, n_test_batches
