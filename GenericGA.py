from AbstractGA import *
from random import random
from random import shuffle

class GenericGA(AbstractGA):

  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens):

    self.POP_SIZE = num
    self.TOURNEY_SIZE = tNum
    self.MUT_PROB = mProb
    self.CROSS_PROB = cProb
    self.TOURNEY_PROB = tProb
    self.RANK_PROB = rProb
    self.ELITE = elite
    self.K = k
    self.SORT_DIR = dir_
    self.MAX_GENS = maxGens

    self.mProb = self.MUT_PROB
    self.cProb = self.CROSS_PROB

    self.pop = {}
    if self.SORT_DIR == 0:
      self.best = 1000000
      self.max = 0
    else:
      self.best = 0
      self.max = 1000000

    self.gens = 0

  def fitFunction(self, chrom):
    pass

  def adaptivemodprob(f):
    def wrapper(*args, **kw):
      try:
        return f(*args, **kw)
      except TypeError:
        self = args[0]
        op = args[-1]
        args = args[:-1]
        avg = sum([self.fitFunction(x) for x in self.pop.keys()])/len(self.pop)
        if op(self.best, avg):
          self.mProb = self.MUT_PROB*(self.max-self.best)/(self.max/avg)
          self.cProb = self.CROSS_PROB*(self.max-self.best)/(self.max/avg)
        else:
          self.mProb = self.MUT_PROB
          self.cProb = self.CROSS_PROB
        return f(*args, **kw)
    return wrapper
  
  @adaptivemodprob
  def run(self, selFunction, mutFunction, crossFunction):
    try:
      matePool, self.best = selFunction()
      self.pop = {}
      if self.ELITE:
        self.pop[self.best] = None
      mateSelector = self.poolGen(matePool)
      mates = mateSelector.next
      while len(self.pop) < self.POP_SIZE:
        p1, p2 = mates()
        if random() < self.cProb:
          c1, c2 = crossFunction(p1, p2)
        else:
          c1, c2 = p1, p2
        c1 = mutFunction(c1)
        c2 = mutFunction(c2)
        self.pop[c1] = None
        self.pop[c2] = None
      del mateSelector
      self.gens += 1
      return self.best, self.gens
    except KeyboardInterrupt:
      pass

  def poolGen(self, matePool):
    half = len(matePool)/2
    while True:
      shuffle(matePool)
      for p1, p2 in zip(matePool[half:], matePool[:half]):
        yield p1, p2

  def rouletteSel(self):
    return super(GenericGA, self).rouletteSel(self.pop.keys(), self.fitFunction,
                                          self.SORT_DIR, self.ELITE)

  def rankSel(self):
    return super(GenericGA, self).rankSel(self.pop.keys(), self.fitFunction,
                                      self.linearRank, self.RANK_PROB,
                                      self.SORT_DIR, self.ELITE)

  def tournamentSel(self):
    return super(GenericGA, self).tournamentSel(self.pop.keys(), self.fitFunction,
                                            self.TOURNEY_SIZE,
                                            self.TOURNEY_PROB, self.SORT_DIR,
                                            self.ELITE)

  def heuristicMutate(self, member):
    return super(GenericGA, self).heuristicMutate(self.K, self.fitFunction,
                                              self.SORT_DIR,
                                              member, self.mProb)

  def insertionMutate(self, member):
    return super(GenericGA, self).insertionMutate(member, self.mProb)

  def inversionMutate(self, member):
    return super(GenericGA, self).inversionMutate(member, self.mProb)

  def reciprocalMutate(self, member):
    return super(GenericGA, self).reciprocalMutate(member, self.mProb)


class GenericGAQ2(GenericGA):
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    super(GenericGAQ2, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                      elite, k, dir_, maxGens)
    self.MIN = min_
    self.MAX = max_
    self.CHROM_SIZE = gSize
    self.NOISE = range(-noise, noise+1)
    self.NOISE.remove(0)

    self.pop = super(GenericGAQ2, self).genPop(self.CHROM_SIZE, self.POP_SIZE,
                                               self.MIN, self.MAX)

  def basicMutate(self, member):
    return super(GenericGAQ2, self).basicMutate(member, self.mProb, self.NOISE,
                                                self.MIN, self.MAX)
