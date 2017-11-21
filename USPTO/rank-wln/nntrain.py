import tensorflow as tf
from utils.nn import linearND, linear
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph
from models import *
import math, sys, random
from optparse import OptionParser
import threading

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-p", "--cand", dest="cand_path", default=None)
parser.add_option("-c", "--ncore", dest="core_size", default=10)
parser.add_option("-a", "--ncand", dest="cand_size", default=500)
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-n", "--max_norm", dest="max_norm", default=100.0)
opts,args = parser.parse_args()

hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
core_size = int(opts.core_size)
cutoff = int(opts.cand_size)
max_norm = float(opts.max_norm)

session = tf.Session()
_input_atom = tf.placeholder(tf.float32, [None, None, adim])
_input_bond = tf.placeholder(tf.float32, [None, None, bdim])
_atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_num_nbs = tf.placeholder(tf.int32, [None, None])
_label = tf.placeholder(tf.int32, [None])
_src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _label]

q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32])
enqueue = q.enqueue(_src_holder)
input_atom, input_bond, atom_graph, bond_graph, num_nbs, label = q.dequeue()

input_atom.set_shape([None, None, adim])
input_bond.set_shape([None, None, bdim])
atom_graph.set_shape([None, None, max_nb, 2])
bond_graph.set_shape([None, None, max_nb, 2])
num_nbs.set_shape([None, None])
label.set_shape([None])

graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs) 
with tf.variable_scope("encoder"):
    _, fp = rcnn_wl_last(graph_inputs, hidden_size=hidden_size, depth=depth)

reactant = fp[0:1,:]
candidates = fp[1:,:]
candidates = candidates - reactant
#reactant = linear(reactant, hidden_size, "reactant", init_bias=None)
candidates = linear(candidates, hidden_size, "candidate")
#match = reactant + candidates
match = tf.nn.relu(candidates)
score = tf.squeeze(linear(match, 1, "score"), [1])
loss = tf.nn.softmax_cross_entropy_with_logits(score, label)
pred = tf.argmax(score, 0)

_lr = tf.placeholder(tf.float32, [])
optimizer = tf.train.AdamOptimizer(learning_rate=_lr)
param_norm = tf.global_norm(tf.trainable_variables())
grads_and_vars = optimizer.compute_gradients(loss) 
grads, var = zip(*grads_and_vars)
grad_norm = tf.global_norm(grads)
new_grads, _ = tf.clip_by_global_norm(grads, max_norm)
grads_and_vars = zip(new_grads, var)
backprop = optimizer.apply_gradients(grads_and_vars)

tf.global_variables_initializer().run(session=session)
size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.trainable_variables())
print "Model size: %dK" % (n/1000,)

def read_data(coord):
    data = []
    train_f = open(opts.train_path, 'r')
    cand_f = open(opts.cand_path, 'r')
        
    for line in train_f:
        r,e = line.strip("\r\n ").split()
        cand = cand_f.readline()

        cbonds = []
        for b in e.split(';'):
            x,y = b.split('-')
            x,y = int(x)-1,int(y)-1
            cbonds.append((x,y))

        sbonds = set(cbonds)

        for b in cand.strip("\r\n ").split():
            x,y = b.split('-')
            x,y = int(x)-1,int(y)-1
            if (x,y) not in sbonds:
                cbonds.append((x,y))
        data.append((r,cbonds))
     
    data_len = len(data)
    it = 0
    while True:
        reaction, cand_bonds = data[it]
        cand_bonds = cand_bonds[:core_size]
        it = (it + 1) % data_len
        r,_,p = reaction.split('>')
        src_tuple,_ = smiles2graph(r, p, cand_bonds, cutoff=cutoff)
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        session.run(enqueue, feed_dict=feed_map)
    coord.request_stop()

coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(coord,))
t.start()

saver = tf.train.Saver()
it, sum_acc, sum_err, sum_gnorm = 0, 0.0, 0.0, 0.0
lr = 0.001
try:
    while not coord.should_stop():
        it += 1
        _, cur_pred, pnorm, gnorm = session.run([backprop, pred, param_norm, grad_norm], feed_dict={_lr:lr})
        if cur_pred != 0: sum_err += 1.0
        sum_gnorm += gnorm

        if it % 200 == 0 and it > 0:
            print "Training Error: %.4f, Param Norm: %.2f, Grad Norm: %.2f" % (sum_err / 200, pnorm, sum_gnorm / 200) 
            sys.stdout.flush()
            sum_err, sum_gnorm = 0.0, 0.0
        if it % 20000 == 0 and it > 0:
            saver.save(session, opts.save_path + "/model.ckpt-%d" % it)
            lr *= 0.9
            print "Learning Rate: %.6f" % lr

except Exception as e:
    coord.request_stop(e)
finally:
    saver.save(session, opts.save_path + "/model.final")
    coord.request_stop()
    coord.join([t])
