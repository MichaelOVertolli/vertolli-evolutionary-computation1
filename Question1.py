import random
import string
import copy
from GenericGA import GenericGAQ2

class StrSearch(GenericGAQ2):

  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize, query):
    super(StrSearch, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                    elite, k, dir_, maxGens, min_, max_, noise,
                                    gSize)
    self.NUM_TO_CHAR = []
    self.NUM_TO_CHAR.append(' ')
    self.NUM_TO_CHAR.extend([x for x in string.ascii_lowercase])
    self.CHAR_TO_NUM = {}
    for num, char in enumerate(self.NUM_TO_CHAR):
      self.CHAR_TO_NUM[char] = num

    self.query = tuple(self.convert(query))
    self.bestVal = 'Nil'

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
    self.best, self.bestVal = selFunction()
    self.pop = {}
    if self.ELITE:
      self.pop[self.best] = None
    while len(self.pop) < self.POP_SIZE:
      c = mutFunction([x for x in self.best])
      self.pop[c] = None
    self.gens += 1
    return self.convert(self.best), self.gens

  def fitFunction(self, chrom):
    """Fake fit function for convenience in GUI."""
    try:
      return sum([abs(x - y) for x, y in zip(chrom, self.query)])
    except TypeError:
      return sum([abs(x - y) for x, y in zip(self.convert(chrom), self.query)])

  def getClosest(self):
    bVal = 1000000
    bestStr = []
    for obj in self.pop.keys():
      val = sum([abs(x - y) for x, y in zip(obj, self.query)])
      if val < bVal:
        bVal = val
        bestStr = obj
    return bestStr, bVal

  def switchRep(self, val):
    #Number to character switch is more common so faster to do it first.
    try:
      return self.NUM_TO_CHAR[val]
    except TypeError:
      return self.CHAR_TO_NUM[val]

  def convert(self, obj):
    newStr = ''
    if type(obj) == tuple:
      newStr = ''.join(map(self.switchRep, list(obj)))
    else:
      newStr = map(self.switchRep, list(obj))
    return newStr
