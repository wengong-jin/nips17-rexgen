gold = open('../data/test.single')
fpop = open('../data/test.pop')
cand = open('test.cbond')

mincnt = [4000, 2000, 1000, 500, 200, 50, 5, -10]
tot = [0] * len(mincnt)
s6 = [0] * len(mincnt)
s8 = [0] * len(mincnt)
s10 = [0] * len(mincnt)

for line in cand:
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

    pop = int(fpop.readline())
    
    for i in xrange(len(mincnt)):
        if pop >= mincnt[i]:
            tot[i] += 1
            if set(gold_bonds) <= set(cand_bonds[:6]):
                s6[i] += 1.0
            if set(gold_bonds) <= set(cand_bonds[:8]):
                s8[i] += 1.0
            if set(gold_bonds) <= set(cand_bonds[:10]):
                s10[i] += 1.0
            break

for i in xrange(len(mincnt)):
    print '%d, %.4f, %.4f, %.4f' % (tot[i], s6[i] / tot[i], s8[i] / tot[i], s10[i] / tot[i])
