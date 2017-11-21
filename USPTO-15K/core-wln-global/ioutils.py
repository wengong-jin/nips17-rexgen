import rdkit.Chem as Chem
from mol_graph import bond_fdim, bond_features
import numpy as np

BOND_TYPE = ["NOBOND", Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
N_BOND_CLASS = len(BOND_TYPE)
binary_fdim = 4 + bond_fdim
INVALID_BOND = -1

def get_bin_feature(r, max_natoms):
    comp = {}
    for i, s in enumerate(r.split('.')):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            comp[atom.GetIntProp('molAtomMapNumber') - 1] = i
    n_comp = len(r.split('.'))
    rmol = Chem.MolFromSmiles(r)
    n_atoms = rmol.GetNumAtoms()
    bond_map = {}
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        bond_map[(a1,a2)] = bond_map[(a2,a1)] = bond
        
    features = []
    for i in xrange(max_natoms):
        for j in xrange(max_natoms):
            f = np.zeros((binary_fdim,))
            if i >= n_atoms or j >= n_atoms or i == j:
                features.append(f)
                continue
            if (i,j) in bond_map:
                bond = bond_map[(i,j)]
                f[1:1+bond_fdim] = bond_features(bond)
            else:
                f[0] = 1.0
            f[-4] = 1.0 if comp[i] != comp[j] else 0.0
            f[-3] = 1.0 if comp[i] == comp[j] else 0.0
            f[-2] = 1.0 if n_comp == 1 else 0.0
            f[-1] = 1.0 if n_comp > 1 else 0.0
            features.append(f)
    return np.vstack(features).reshape((max_natoms,max_natoms,binary_fdim))

def get_bond_label(r, edits, max_natoms):
    rmol = Chem.MolFromSmiles(r)
    n_atoms = rmol.GetNumAtoms()
    rmap = np.zeros((max_natoms, max_natoms))

    delbond = edits.split(';')[2]
    newbond = edits.split(';')[3]
    if len(delbond) > 0:
        for s in delbond.split(','):
            x,y,_ = s.split('-')
            x,y = int(x)-1,int(y)-1
            rmap[x,y] = rmap[y,x] = 1
    if len(newbond) > 0:
        for s in newbond.split(','):
            x,y,_ = s.split('-')
            x,y = int(x)-1,int(y)-1
            rmap[x,y] = rmap[y,x] = 1

    labels = []
    sp_labels = []
    for i in xrange(max_natoms):
        for j in xrange(max_natoms):
            if i == j or i >= n_atoms or j >= n_atoms:
                labels.append(INVALID_BOND)
            else:
                labels.append(rmap[i,j])
                if rmap[i,j] == 1:
                    sp_labels.append(i * max_natoms + j)
    return np.array(labels), sp_labels

def get_all_batch(re_list):
    mol_list = []
    max_natoms = 0
    for r,e in re_list:
        rmol = Chem.MolFromSmiles(r)
        mol_list.append((r,e))
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()
    labels = []
    features = []
    sp_labels = []
    for r,e in mol_list:
        l, sl = get_bond_label(r,e,max_natoms)
        features.append(get_bin_feature(r,max_natoms))
        labels.append(l)
        sp_labels.append(sl)
    return np.array(features), np.array(labels), sp_labels
