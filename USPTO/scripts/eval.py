import rdkit
from rdkit import Chem
from optparse import OptionParser

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)

BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def edit_mol(rmol, edits):
    new_mol = copy_edit_mol(rmol)
    amap = {}
    for atom in rmol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber')] = atom.GetIdx()

    for x,y,t in edits:
        bond = new_mol.GetBondBetweenAtoms(amap[x],amap[y])
        a1 = new_mol.GetAtomWithIdx(amap[x])
        a2 = new_mol.GetAtomWithIdx(amap[y])
        if bond is not None:
            new_mol.RemoveBond(amap[x],amap[y])
        if t > 0:
            new_mol.AddBond(amap[x],amap[y],BOND_TYPE[t])

    pred_mol = new_mol.GetMol()

    #Clear formal charges to make molecules valid
    for atom in pred_mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        if atom.GetSymbol() == 'O' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ['Cl', 'I', 'P', 'C'] and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)

    pred_smiles = Chem.MolToSmiles(pred_mol)
    pred_list = pred_smiles.split('.')
    pred_mols = [Chem.MolFromSmiles(pred_smiles) for pred_smiles in pred_list]
    pred_smiles = [Chem.MolToSmiles(pred_mol) for pred_mol in pred_mols if pred_mol is not None]

    return pred_smiles

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-t", "--pred", dest="pred_path")
    parser.add_option("-g", "--gold", dest="gold_path")
    opts,args = parser.parse_args()

    fpred = open(opts.pred_path)
    fgold = open(opts.gold_path)

    rank = []
    n,top1,top3,top5 = 0,0,0,0
    for line in fpred:
        line = line.strip('\r\n |')
        gold = fgold.readline()
        rex,_ = gold.split()
        r,_,p = rex.split('>')

        rmol = Chem.MolFromSmiles(r)
        pmol = Chem.MolFromSmiles(p)

        for atom in pmol.GetAtoms():
            atom.ClearProp('molAtomMapNumber')

        psmiles = Chem.MolToSmiles(pmol)
        psmiles = set(psmiles.split('.'))
       
        rk = 10
        for idx,edits in enumerate(line.split('|')):
            cbonds = [] 
            for edit in edits.split():
                x,y,t = edit.split('-')
                x,y,t = int(x),int(y),int(t)
                cbonds.append((x,y,t))
                
            #Generate products by modifying reactants with predicted edits.
            pred_smiles = edit_mol(rmol, cbonds)
            pred_smiles = set(pred_smiles)
            if psmiles <= pred_smiles:
                rk = idx + 1
                break

            #Edit molecules with reactants kekulized. Sometimes previous editing fails due to RDKit sanitization error (edited molecule cannot be kekulized)
            try:
                Chem.Kekulize(rmol)
            except Exception as e:
                pass

            pred_smiles = edit_mol(rmol, cbonds)
            pred_smiles = set(pred_smiles)
            if psmiles <= pred_smiles:
                rk = idx + 1
                break
            
        n += 1.0
        if rk == 1: top1 += 1
        if rk <= 3: top3 += 1
        if rk <= 5: top5 += 1

        print '%.4f, %.4f, %.4f' % (top1 / n, top3 / n, top5 / n)
