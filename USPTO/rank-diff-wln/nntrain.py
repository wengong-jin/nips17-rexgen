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
parser.add_option("-b", "--batch", dest="batch_size", default=4)
parser.add_option("-c", "--ncore", dest="core_size", default=10)
parser.add_option("-a", "--ncand", dest="cand_size", default=500)
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-n", "--max_norm", dest="max_norm", default=50.0)
opts,args = parser.parse_args()

hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
core_size = int(opts.core_size)
cutoff = int(opts.cand_size)
max_norm = float(opts.max_norm)
batch_size = int(opts.batch_size)

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

with tf.variable_scope("mol_encoder"):
    fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

reactant = fp_all_atoms[0:1,:]
candidates = fp_all_atoms[1:,:]
candidates = candidates - reactant
candidates = tf.concat(0, [reactant, candidates])

with tf.variable_scope("diff_encoder"):
    reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=1)

reaction_fp = reaction_fp[1:]
reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

score = tf.squeeze(linear(reaction_fp, 1, "score"), [1])
loss = tf.nn.softmax_cross_entropy_with_logits(score, label)
pred = tf.argmax(score, 0)

_lr = tf.placeholder(tf.float32, [])
optimizer = tf.train.AdamOptimizer(learning_rate=_lr)

tvs = tf.trainable_variables()
param_norm = tf.global_norm(tvs)

grads_and_vars = optimizer.compute_gradients(loss, tvs) 
grads, var = zip(*grads_and_vars)
grad_norm = tf.global_norm(grads)
new_grads, _ = tf.clip_by_global_norm(grads, max_norm)

accum_grads = [tf.Variable(tf.zeros(v.get_shape().as_list()), trainable=False) for v in tvs]
zero_ops = [v.assign(tf.zeros(v.get_shape().as_list())) for v in accum_grads]
accum_ops = [accum_grads[i].assign_add(grad) for i, grad in enumerate(new_grads)]

grads_and_vars = zip(accum_grads, var)
backprop = optimizer.apply_gradients(grads_and_vars)

tf.global_variables_initializer().run(session=session)

size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.trainable_variables())
print "Model size: %dK" % (n/1000,)

def count(s):
    c = 0
    for i in xrange(len(s)):
        if s[i] == ':':
            c += 1
    return c

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

    random.shuffle(data)
    data_len = len(data)
    it = 0
    while True:
        reaction, cand_bonds = data[it]
        cand_bonds = cand_bonds[:core_size]
        it = (it + 1) % data_len
        r,_,p = reaction.split('>')
        n = count(r)
        if n <= 2 or n > 100: continue
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
        it += batch_size
        session.run(zero_ops)
        for i in xrange(batch_size):
            ans = session.run(accum_ops + [pred])
            if ans[-1] != 0: 
                sum_err += 1.0

        _, pnorm, gnorm = session.run([backprop, param_norm, grad_norm], feed_dict={_lr:lr})
        sum_gnorm += gnorm

        if it % 200 == 0 and it > 0:
            print "Training Error: %.4f, Param Norm: %.2f, Grad Norm: %.2f" % (sum_err / 200, pnorm, sum_gnorm / 200 * batch_size) 
            sys.stdout.flush()
            sum_err, sum_gnorm = 0.0, 0.0
        if it % 40000 == 0 and it > 0:
            saver.save(session, opts.save_path + "/model.ckpt-%d" % it)
            lr *= 0.9
            print "Learning Rate: %.6f" % lr

except Exception as e:
    print e
    coord.request_stop(e)
finally:
    saver.save(session, opts.save_path + "/model.final")
    coord.request_stop()
    coord.join([t])
