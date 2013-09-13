from GenericGA import GenericGA
from math import sqrt

class TSPGA(GenericGA):

  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, file_):

    super(TSPGA, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                elite, k, dir_, maxGens)

    self.MAPPED = self.load(file_)

    self.pop = self.genPopPermut(self.MAPPED.keys(), self.POP_SIZE)

  def load(self, file_):
    mapper = {}
    with open(file_, 'r') as f:
      for line in f.readlines():
        line = line.strip()
        line = line.split()
        try:
          mapper[int(line[0])] = (float(line[1]), float(line[2]))
        except IndexError:
          pass
    return mapper

  def fitFunction(self, chrom):
    gen = self.nextTwo(chrom)
    total = 0
    for pair in list(gen):
      total += self.euclidian(self.MAPPED[pair[0]], self.MAPPED[pair[1]])
    return total

  def euclidian(self, point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

  def prep(self, chrom, divider):
    gen = self.nextTwo(chrom)
    gen = list(gen)
    gen = [(self.MAPPED[pair[0]], self.MAPPED[pair[1]]) for pair in gen]
    gen = [[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] for pair in gen]
    for pos, x in enumerate(gen):
      for pos2, y in enumerate(x):
        gen[pos][pos2] = gen[pos][pos2]/divider
    return gen
    
class TSPGA1(TSPGA):
  def __init__(self):
    super(TSPGA1, self).__init__(500, 10, 0.001, 0.7, 0.6, False, 3, 0,
                                 2000, 'berlin52.txt')
  def go(self):
    return self.run(self.tournamentSel, self.heuristicMutate,
                    self.edgeRecombUI)

