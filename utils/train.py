import tensorflow as tf
from tensorflow.python.client import device_lib

def cross_entropy(logits, labels):
    return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def weight_decay(decay, var_list=None):
    var_list = tf.trainable_variables() if var_list is None else var_list
    return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def get_train_op(optim, loss, global_step=None, clip=None, var_list=None):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grad_and_vars = optim.compute_gradients(loss, var_list=var_list)
        if clip is not None:
            grad_and_vars = [((None if grad is None \
                    else tf.clip_by_norm(grad, clip)), var) \
                    for grad, var in grad_and_vars]
        train_op = optim.apply_gradients(grad_and_vars, global_step=global_step)
        return train_op

# copied from https://stackoverflow.com/a/38580201
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    mem_thres = 0.3*max([x.memory_limit for x in local_device_protos \
            if x.device_type=='GPU'])
    return [x.name for x in local_device_protos if x.device_type=='GPU' \
            and x.memory_limit > mem_thres]

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
