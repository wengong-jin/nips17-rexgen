import tensorflow as tf
from utils.nn import linearND, linear
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from models import *
from ioutils import *
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading, pickle
from multiprocessing import Queue


parser = OptionParser()
parser.add_option("-t", "--data", dest="data_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-n", "--ncand", dest="ncand", default=20)
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
NK = int(opts.ncand)

smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

gpu_options = tf.GPUOptions()
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
_input_atom = tf.placeholder(tf.float32, [batch_size, None, adim])
_input_bond = tf.placeholder(tf.float32, [batch_size, None, bdim])
_atom_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
_bond_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
_num_nbs = tf.placeholder(tf.int32, [batch_size, None])
_node_mask = tf.placeholder(tf.float32, [batch_size, None])
_src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask]
_label = tf.placeholder(tf.int32, [batch_size, None])
_binary = tf.placeholder(tf.float32, [batch_size, None, None, binary_fdim])
#keep_prob = tf.placeholder(tf.float32)

q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.float32])
enqueue = q.enqueue(_src_holder + [_label, _binary])
input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, label, binary = q.dequeue()

input_atom.set_shape([batch_size, None, adim])
input_bond.set_shape([batch_size, None, bdim])
atom_graph.set_shape([batch_size, None, max_nb, 2])
bond_graph.set_shape([batch_size, None, max_nb, 2])
num_nbs.set_shape([batch_size, None])
node_mask.set_shape([batch_size, None])
label.set_shape([batch_size, None])
binary.set_shape([batch_size, None, None, binary_fdim])

node_mask = tf.expand_dims(node_mask, -1)

graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)
with tf.variable_scope("encoder"):
    atom_hiddens, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

atom_hiddens1 = tf.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
atom_hiddens2 = tf.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
atom_pair = atom_hiddens1 + atom_hiddens2

att_hidden = tf.nn.relu(linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="att_bin_feature"))
att_score = linearND(att_hidden, 1, scope="att_scores")
att_score = tf.nn.sigmoid(att_score)
att_context = att_score * atom_hiddens1
att_context = tf.reduce_sum(att_context, 2)

att_context1 = tf.reshape(att_context, [batch_size, 1, -1, hidden_size])
att_context2 = tf.reshape(att_context, [batch_size, -1, 1, hidden_size])
att_pair = att_context1 + att_context2

pair_hidden = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="bin_feature", init_bias=None) + linearND(att_pair, hidden_size, scope="ctx_feature")
pair_hidden = tf.nn.relu(pair_hidden)
pair_hidden = tf.reshape(pair_hidden, [batch_size, -1, hidden_size])

score = linearND(pair_hidden, 1, scope="scores")
score = tf.squeeze(score, [2])
bmask = tf.to_float(tf.equal(label, INVALID_BOND)) * 10000
label_dim = tf.shape(label)[1]
topk_value, topk = tf.nn.top_k(tf.nn.sigmoid(score - bmask), k=NK)

tf.global_variables_initializer().run(session=session)
saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint(opts.model_path))

queue = Queue()

data = []
with open(opts.data_path) as f:
    for line in f:
        items = line.split()
        #if items[1] != 'test': continue
        react = items[0].split('>')[0]
        edits = items[2]
        data.append((react, edits))

def read_data(data, coord):
    it = 0
    for it in xrange(0, len(data), batch_size):
        src_batch, edit_batch = [], []
        for i in xrange(batch_size):
            react = data[it+i][0]
            edits = data[it+i][1]
            src_batch.append(react)
            edit_batch.append(edits)
        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        feed_map.update({_label:cur_label, _binary:cur_bin})
        session.run(enqueue, feed_dict=feed_map)
        queue.put(sp_label)

    coord.request_stop()

coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(data, coord))
t.start()

it, sum_acc, sum_err = 0, 0.0, 0.0
try:
    while not coord.should_stop():
        cur_value, cur_topk, cur_dim = session.run([topk_value, topk, label_dim])
        percentage = np.cumsum(cur_value, -1) / np.sum(cur_value, -1, keepdims=True)
        cur_dim = int(math.sqrt(cur_dim))
        sp_label = queue.get()
        for i in xrange(batch_size):
            pre,rec = 0,0
            for j in xrange(NK):
                k = cur_topk[i,j]
                x = k / cur_dim + 1
                y = k % cur_dim + 1
                if x < y:
                    print "%d-%d" % (x, y),
                if cur_topk[i,j] in sp_label[i]:
                    pre += 1
            print
            if len(sp_label[i]) == pre: 
                sum_acc += 1
    #print "Acc: %.4f" % (sum_acc / 3000,)
except Exception as e:
    print e
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join([t])
