# Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

This repository contains the data and codes of the paper:

*Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network (NIPS 2017)* [PDF](https://arxiv.org/abs/1709.04555)

## Data
* USPTO-15K/data.zip contains the train/dev/test split of USPTO-15K dataset used in our paper, borrowed from [Coley et al.](http://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00064) Each line in the file has two fields, separated by space:
  * Reaction smiles (products are not atom mapped)
  * Four types of reaction edits: 
    1. Atoms obtained Hydrogens
    2. Atoms lost Hydrogens
    3. Deleted bonds
    4. Added bonds
    
    The first two subfields contains a list of atom. The last two subfields includes a list of triples in the form of (atom1-atom2-bondtype). All atoms are indicated by their atom map numbers given in the reaction smiles. See the following table for all possible bond types:
    
  | Bond Type | Bond Name    |
  | ------    |:------------:|
  | 1.0       | Single bond  |
  | 2.0       | Double bond  |
  | 3.0       | Triple bond  |
  | 1.5       | Aromatic bond|

* USPTO/data.zip includes the train/dev/test split of USPTO dataset used in our paper. It has in total 480K fully atom mapped reactions. Each line in the file has two fields, separated by space:
  * Reaction smiles (both reactants and products are atom mapped)
  * Reaction center. That is, atom pairs whose bonds in between changed in the reaction. Atoms are represented by their atom map number given in the reaction smiles.
  
## Codes
Coming soon.

## Contributors
Wengong Jin (wengong@csail.mit.edu)

Connor W. Coley (ccoley@mit.edu)
