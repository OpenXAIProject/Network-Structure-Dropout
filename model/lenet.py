from layers import *
from utils.train import *

def lenet_dense(x, y, training, name='lenet', reuse=None,
        dropout=None, **dropout_kwargs):
    dropout_ = lambda x, subname: x if dropout is None else \
            dropout(x, training, name=name+subname, reuse=reuse,
                    **dropout_kwargs)
    x = dense(dropout_(x, '/dropout1'), 500, activation=relu,
            name=name+'/dense1', reuse=reuse)
    x = dense(dropout_(x, '/dropout2'), 300, activation=relu,
            name=name+'/dense2', reuse=reuse)
    x = dense(dropout_(x, '/dropout3'), 10, name=name+'/dense3', reuse=reuse)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['qpi_vars'] = [v for v in all_vars if 'qpi_vars' in v.name]
    net['pzx_vars'] = [v for v in all_vars if 'pzx_vars' in v.name]
    net['weights'] = [v for v in all_vars \
            if 'qpi_vars' not in v.name and 'pzx_vars' not in v.name]

    net['cent'] = cross_entropy(x, y)
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(x, y)

    prefix = 'train_' if training else 'test_'
    net['kl'] = tf.get_collection('kl')
    net['pi'] = tf.get_collection(prefix+'pi')
    net['n_active'] = tf.get_collection(prefix+'n_active')

    return net

def lenet_conv(x, y, training, name='lenet', reuse=None,
        dropout=None, **dropout_kwargs):
    dropout_ = lambda x, subname: x if dropout is None else \
            dropout(x, training, name=name+subname, reuse=reuse,
                    **dropout_kwargs)
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = conv(x, 20, 5, name=name+'/conv1', reuse=reuse)
    x = relu(dropout_(x, '/dropout1'))
    x = pool(x, name=name+'/pool1')
    x = conv(x, 50, 5, name=name+'/conv2', reuse=reuse)
    x = relu(dropout_(x, '/dropout2'))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(dropout_(x, '/dropout3'), 500, activation=relu,
            name=name+'/dense1', reuse=reuse)
    x = dense(dropout_(x, '/dropout4'), 10, name=name+'/dense2', reuse=reuse)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['qpi_vars'] = [v for v in all_vars if 'qpi_vars' in v.name]
    net['pzx_vars'] = [v for v in all_vars if 'pzx_vars' in v.name]
    net['weights'] = [v for v in all_vars \
            if 'qpi_vars' not in v.name and 'pzx_vars' not in v.name]

    net['cent'] = cross_entropy(x, y)
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(x, y)

    prefix = 'train_' if training else 'test_'
    net['kl'] = tf.get_collection('kl')
    net['pi'] = tf.get_collection(prefix+'pi')
    net['n_active'] = tf.get_collection(prefix+'n_active')

    return net
