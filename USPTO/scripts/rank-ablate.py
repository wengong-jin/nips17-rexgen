gold = open('../data/test.pop')
rank = open('../rank-diff-wln/test.rank')

mincnt = [4000, 2000, 1000, 500, 200, 50, 5, -10]
tot = [0] * len(mincnt)
mrr = [0] * len(mincnt)

for line in rank:
    rk = int(line)
    pop = int(gold.readline())
    for i in xrange(len(mincnt)):
        if pop >= mincnt[i]:
            tot[i] += 1
            if rk < 10: mrr[i] += 1.0 / rk
            break

for i in xrange(len(mincnt)):
    print tot[i], mrr[i] / tot[i]

