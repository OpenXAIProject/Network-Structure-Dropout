from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from model.lenet import lenet_dense
from model.bbdropout import bbdropout
from utils.accumulator import Accumulator
from utils.train import *
from utils.mnist import mnist_input
import time
import os
import argparse
import csv
from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--pretraindir', type=str, default=None)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--csvfn', type=str, default=None)
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

pretraindir = './results/pretrained' if args.pretraindir is None else args.pretraindir
savedir = './results/bbdropout/sample_run' if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

batch_size = args.batch_size
mnist, n_train_batches, n_test_batches = mnist_input(batch_size)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
N = mnist.train.num_examples
dropout = bbdropout
net = lenet_dense(x, y, True, dropout=dropout)
tnet = lenet_dense(x, y, False, reuse=True, dropout=dropout)

def train():
    loss = net['cent'] + tf.add_n(net['kl'])/float(N) + net['wd']
    global_step = tf.train.get_or_create_global_step()
    bdr = [int(n_train_batches*(args.n_epochs-1)*r) for r in [0.5, 0.75]]
    vals = [1e-2, 1e-3, 1e-4]
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), bdr, vals)
    train_op1 = tf.train.AdamOptimizer(lr).minimize(loss,
            var_list=net['qpi_vars'], global_step=global_step)
    train_op2 = tf.train.AdamOptimizer(0.1*lr).minimize(loss,
            var_list=net['weights'])
    train_op = tf.group(train_op1, train_op2)

    pretrain_saver = tf.train.Saver(net['weights'])
    saver = tf.train.Saver(net['weights']+net['qpi_vars'])
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pretrain_saver.restore(sess, os.path.join(pretraindir, 'model'))

    train_logger = Accumulator('cent', 'acc')
    train_to_run = [train_op, net['cent'], net['acc']]
    test_logger = Accumulator('cent', 'acc')
    test_to_run = [tnet['cent'], tnet['acc']]
    for i in range(args.n_epochs):
        line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
        print(line)
        logfile.write(line + '\n')
        train_logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            bx, by = mnist.train.next_batch(batch_size)
            train_logger.accum(sess.run(train_to_run, {x:bx, y:by}))
        train_logger.print_(header='train', epoch=i+1,
                time=time.time()-start, logfile=logfile)

        test_logger.clear()
        for j in range(n_test_batches):
            bx, by = mnist.test.next_batch(batch_size)
            test_logger.accum(sess.run(test_to_run, {x:bx, y:by}))
        test_logger.print_(header='test', epoch=i+1,
                time=time.time()-start, logfile=logfile)
        line = 'kl: ' + str(sess.run(tnet['kl'])) + '\n'
        line += 'n_active: ' + str(sess.run(tnet['n_active'])) + '\n'
        print(line)
        logfile.write(line+'\n')

        if (i+1)%args.save_freq == 0:
            saver.save(sess, os.path.join(savedir, 'model'))

    logfile.close()
    saver.save(sess, os.path.join(savedir, 'model'))

def test():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights']+tnet['qpi_vars'])
    saver.restore(sess, os.path.join(savedir, 'model'))
    logger = Accumulator('cent', 'acc')
    to_run = [tnet['cent'], tnet['acc']]
    for j in range(n_test_batches):
        bx, by = mnist.test.next_batch(batch_size)
        logger.accum(sess.run(to_run, {x:bx, y:by}))
    logger.print_(header='test')
    line = 'kl: ' + str(sess.run(tnet['kl'])) + '\n'
    line += 'n_active: ' + str(sess.run(tnet['n_active'])) + '\n'
    print(line)

def visualize():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights']+tnet['qpi_vars'])
    saver.restore(sess, os.path.join(savedir, 'model'))

    n_drop = len(tnet['n_active'])
    fig = figure('pi')
    axarr = fig.subplots(n_drop)
    for i in range(n_drop):
        np_pi = sess.run(tnet['pi'][i]).reshape((1,-1))
        im = axarr[i].imshow(np_pi, cmap='gray', aspect='auto')
        axarr[i].yaxis.set_visible(False)
        axarr[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        if i == n_drop-1:
            axarr[i].set_xlabel('neurons')
        fig.colorbar(im, ax=axarr[i])
    show()

def record():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights']+tnet['qpi_vars'])
    saver.restore(sess, os.path.join(savedir, 'model'))
    logger = Accumulator('cent', 'acc')
    to_run = [tnet['cent'], tnet['acc']]
    for j in range(n_test_batches):
        bx, by = mnist.test.next_batch(batch_size)
        logger.accum(sess.run(to_run, {x:bx, y:by}))
    np_n_active = sess.run(tnet['n_active'])

    if not os.path.isdir('../../records'):
        os.makedirs('../../records')
    csvfn = os.path.join('../../records',
            'bbdropout_lenet_dense.csv' if args.csvfn is None else args.csvfn)

    if csvfn is not None:
        flag = 'a' if os.path.exists(csvfn) else 'w'
        with open(csvfn, flag) as f:
            writer = csv.writer(f)
            if flag=='w':
                writer.writerow(['savedir', 'cent', 'acc', 'n_active'])
            line = [savedir]
            line.append('%.4f' % logger.get('cent'))
            line.append('%.4f' % logger.get('acc'))
            line.append('-'.join(str(x) for x in np_n_active))
            writer.writerow(line)

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'vis':
        visualize()
    elif args.mode == 'record':
        record()
    else:
        raise ValueError('Invalid mode %s' % args.mode)
