import tensorflow as tf
from utils.nn import linearND, linear
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, smiles2graph_test, bond_types
from models import *
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue
import rdkit
from rdkit import Chem

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-p", "--cand", dest="cand_path", default=None)
parser.add_option("-a", "--ncand", dest="cand_size", default=500)
parser.add_option("-c", "--ncore", dest="core_size", default=10)
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
opts,args = parser.parse_args()

TOPK = 5

hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
core_size = int(opts.core_size)
MAX_NCAND = int(opts.cand_size)

#gpu_options = tf.GPUOptions(allow_growth=True)
#session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
session = tf.Session()
_input_atom = tf.placeholder(tf.float32, [None, None, adim])
_input_bond = tf.placeholder(tf.float32, [None, None, bdim])
_atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_num_nbs = tf.placeholder(tf.int32, [None, None])
_src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs]

q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32])
enqueue = q.enqueue(_src_holder)
input_atom, input_bond, atom_graph, bond_graph, num_nbs = q.dequeue()

input_atom.set_shape([None, None, adim])
input_bond.set_shape([None, None, bdim])
atom_graph.set_shape([None, None, max_nb, 2])
bond_graph.set_shape([None, None, max_nb, 2])
num_nbs.set_shape([None, None])

graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs) 
with tf.variable_scope("mol_encoder"):
    fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

reactant = fp_all_atoms[0:1,:]
candidates = fp_all_atoms[1:,:]
candidates = candidates - reactant
candidates = tf.concat(0, [reactant, candidates])

with tf.variable_scope("diff_encoder"):
    reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=depth)

reaction_fp = reaction_fp[1:]
reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

score = tf.squeeze(linear(reaction_fp, 1, "score"), [1])

tk = tf.minimum(TOPK, tf.shape(score)[0])
_, pred_topk = tf.nn.top_k(score, tk)

tf.global_variables_initializer().run(session=session)

queue = Queue()

def read_data(coord):
    data = []
    data_f = open(opts.test_path, 'r')
    cand_f = open(opts.cand_path, 'r')
    
    for line in data_f:
        items = line.split()
        cand = cand_f.readline()
        r = items[0]
        edits = items[2]

        cand_bonds = []
        for b in cand.strip("\r\n ").split():
            x,y = b.split('-')
            x,y = int(x)-1,int(y)-1
            cand_bonds.append((x,y))

        data.append((r,cand_bonds))

    data_len = len(data)
    for it in xrange(data_len):
        r,cand_bonds = data[it]
        r = r.split('>')[0]
        ncore = core_size
        while True:
            src_tuple,conf = smiles2graph_test(r, cand_bonds[:ncore])
            if len(conf) <= MAX_NCAND:
                break
            ncore -= 1
        queue.put((r,conf))
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        session.run(enqueue, feed_dict=feed_map)
      
    
coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(coord,))
t.start()

saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint(opts.model_path))
total = 0
idxfunc = lambda x:x.GetIntProp('molAtomMapNumber')
try:
    while not coord.should_stop():
        total += 1
        r,conf = queue.get()
        cur_pred = session.run(pred_topk)
        rmol = Chem.MolFromSmiles(r)
        rbonds = {}
        for bond in rmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1,a2 = min(a1,a2),max(a1,a2)
            rbonds[(a1,a2)] = t

        res = conf[cur_pred[0]]
        for idx in cur_pred:
            for x,y,t in conf[idx]:
                x,y = x+1,y+1
                if ((x,y) not in rbonds and t > 0) or ((x,y) in rbonds and rbonds[(x,y)] != t):
                    print '%d-%d-%d' % (x,y,t),
            print '|',
        print
        if total % 1000 == 0:
            sys.stdout.flush()
        
except Exception as e:
    print e
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join([t])
