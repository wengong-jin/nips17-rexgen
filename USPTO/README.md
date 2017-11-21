## Reaction Core Identification

Reaction core identification codes are in core-wln-global. To train model:
```
python nntrain.py --train ../data/train.txt --hidden $HIDDEN --depth $DEPTH --save_dir $MODELDIR
```

To test model:
```
python nntest.py --test ../data/test.txt --hidden $HIDDEN --depth $DEPTH --model $MODELDIR > test.cbond
```
it prints top 10 atom pairs in the reaction center for each reaction. Here `$MODELDIR` refers to folder `core-wln-global/core-300-3`

Here train.cbond and test.cbond are outputs of nntest.py for train.txt and test.txt respectively.

## Candidate Ranking

Codes are in rank-wln and rank-diff-wln. Note that you need to finish training/testing reaction core identification before candidate ranking.
To train model:
```
python nntrain.py --train ../data/train.txt --cand ../core-wln-global/train.cbond --hidden $HIDDEN --depth $DEPTH --ncand 2000 --ncore 8 --save_dir $MODELDIR
```
Here `--cand` argument takes the file of predicted reaction centers from the previous pipeline. 
`--ncore` sets the limit of reaction center size used for candidate generation.
`--ncand` sets the limit of number of candidate products for each reaction.

To test model:
```
python nntest.py --test ../data/train.txt --cand ../core-wln-global/test.cbond --hidden $HIDDEN --depth $DEPTH --ncand 2000 --ncore 8 --save_dir $MODELDIR > test.cbond
```
This outputs top 5 candidate products in one line for each reaction. Note that this script only outputs the bond type assignment
over each atom pair (single/double/triple/delete, etc). You need to run the next script that generates SMILES string of the product.

Here `$MODELDIR` refers to folder `rank-wln/core8-320-3` (hidden=320,depth=3,ncore=8) or `rank-diff-wln/core8-300-2`(hidden=300,depth=2,ncore=8)

For evaluation:
```
python scripts/eval.py --gold ../data/test.txt --pred rand-diff-wln/test.cbond
```
This script internally generates SMILES for each candidate product (given bond assignments to each atom pair), and test if it matches the labeled product.
A candidate product (a set of molecules) is said to match the labeled product (again, a set of molecules) whhen the labeled product is a subset of candidate product.

In addition, we also provided `oracletest.py` that makes prediction with augmented setting. That is, the true product is added to candidate list when it is missing due to errors of reaction center prediction.
