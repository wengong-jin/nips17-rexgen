cand = open('lbp-bond/data.cbond')
gold = open('data/connor_split.csv')

tot,good = 0,0
for line in cand:
    cand_bonds = []
    for v in line.split():
        x,y = v.split('-')
        cand_bonds.append((int(x),int(y)))
    
    line = gold.readline()
    items = line.split()
    if items[1] != 'test': continue

    tot += 1

    edits = items[2]
    gold_bonds = []
    delbond = edits.split(';')[2]
    newbond = edits.split(';')[3]
    if len(delbond) > 0:
        for s in delbond.split(','):
            x,y,_ = s.split('-')
            x,y = int(x),int(y)
            x,y = min(x,y),max(x,y)
            gold_bonds.append((x,y))
    if len(newbond) > 0:
        for s in newbond.split(','):
            x,y,_ = s.split('-')
            x,y = int(x),int(y)
            x,y = min(x,y),max(x,y)
            gold_bonds.append((x,y))

    if set(gold_bonds) <= set(cand_bonds[:8]):
        good += 1.0
print good / tot
