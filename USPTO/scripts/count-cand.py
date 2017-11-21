import rdkit
import rdkit.Chem as Chem
import numpy as np
import sys

bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

def search(buf, cur_bonds, core_bonds, free_vals, depth):
    if depth >= len(core_bonds):
        buf.append([u for u in cur_bonds])
        return
    x,y,t = core_bonds[depth]
    if t >= 0:
        cur_bonds.append((x,y,t))
        free_vals[x] -= t
        free_vals[y] -= t
        search(buf, cur_bonds, core_bonds, free_vals, depth + 1)
        free_vals[x] += t
        free_vals[y] += t
        cur_bonds.pop()
    else:
        for k in xrange(4):
            if k > free_vals[x] or k > free_vals[y]:
                break
            cur_bonds.append((x,y,k))
            free_vals[x] -= k
            free_vals[y] -= k
            search(buf, cur_bonds, core_bonds, free_vals, depth + 1)
            free_vals[x] += k
            free_vals[y] += k
            cur_bonds.pop()
            
def floodfill(cur_id, cur_label, comp, core_bonds):
    comp[cur_id] = cur_label
    x,y = core_bonds[cur_id]
    for i in xrange(len(core_bonds)):
        if comp[i] >= 0: continue
        u,v = core_bonds[i]
        if x == u or x == v or y == u or y == v:
            floodfill(i, cur_label, comp, core_bonds)

def smiles2graph_test(rsmiles, psmiles, core_bonds, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1):
    mol = Chem.MolFromSmiles(rsmiles)
    pmol = Chem.MolFromSmiles(psmiles)
    if not mol or not pmol:
        raise ValueError("Could not parse smiles string:", rsmiles + '>>' + psmiles)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    free_vals = np.zeros((n_atoms,))

    gbonds = {(x,y):0 for x,y in core_bonds}

    #Feature Extraction
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        free_vals[idx] += atom.GetTotalNumHs() + abs(atom.GetFormalCharge())
    
    tatoms = set()
    #Calculate free slots for each atom in product
    for bond in pmol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        t = bond_types.index(bond.GetBondType()) + 1
        a1,a2 = min(a1,a2),max(a1,a2)
        tatoms.add(a1)
        tatoms.add(a2)
        if (a1,a2) in core_bonds:
            gbonds[(a1,a2)] = t
    
    rbonds = {}
    #Calculate free slots for each atom in reactant
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        t = bond_types.index(bond.GetBondType())
        a1,a2 = min(a1,a2),max(a1,a2)
        tval = t + 1 if t < 3 else 1.5
        rbonds[(a1,a2)] = t + 1
        if (a1,a2) in core_bonds:
            free_vals[a1] += tval 
            free_vals[a2] += tval

    #Fix golden label
    for x,y in gbonds.iterkeys():
        if x not in tatoms and y not in tatoms and (x,y) in rbonds:
            gbonds[(x,y)] = rbonds[(x,y)]

    #Gather the golden label
    gold_bonds = set()
    for x,y in core_bonds:
        t = gbonds[(x,y)]
        gold_bonds.add( (x,y,t) )

    #Get connected components in core bonds
    comp = [-1] * len(core_bonds)
    tot = 0
    for i in xrange(len(core_bonds)):
        if comp[i] == -1:
            floodfill(i, tot, comp, core_bonds)
            tot += 1
    
    core_configs = []
    for cur_id in xrange(tot):
        cand_bonds = []
        for i in xrange(len(core_bonds)):
            x,y = core_bonds[i]
            if comp[i] == cur_id: t = -1
            elif (x,y) not in rbonds: t = 0
            else: t = rbonds[(x,y)]
            cand_bonds.append((x,y,t))
        search(core_configs, [], cand_bonds, free_vals, 0)
    
    return len(core_configs)

if __name__ == "__main__":
    train_f = open("data/test.single", 'r')
    cand_f = open("att-rcnn-bond/test.cbond",'r')
    tot = 0.0
    for line in train_f:
        r,e = line.strip("\r\n ").split()
        cand = cand_f.readline()
        cbonds = []

        for b in cand.strip("\r\n ").split():
            x,y = b.split('-')
            x,y = int(x)-1,int(y)-1
            cbonds.append((x,y))

        r,_,p = r.split('>')
        x = smiles2graph_test(r, p, cbonds[:10])
        tot += x

    print tot / 40000
