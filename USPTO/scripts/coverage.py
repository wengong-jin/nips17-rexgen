#cand = open('rcnn-bond/model-300-3/test.cbond')
cand = open('att-rcnn-bond/test.cbond')
gold = open('data/test.single')

tot,good = 0,0
for line in cand:
    tot += 1
    cand_bonds = []
    for v in line.split():
        x,y = v.split('-')
        cand_bonds.append((int(x),int(y)))
    
    line = gold.readline()
    tmp = line.split()[1]
    gold_bonds = []
    for v in tmp.split(';'):
        x,y = v.split('-')
        gold_bonds.append((int(x),int(y)))
    if set(gold_bonds) <= set(cand_bonds[:6]):
        good += 1.0
print good / tot
