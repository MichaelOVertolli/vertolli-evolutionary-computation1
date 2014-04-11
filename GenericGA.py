###############################################################################
#Copyright (C) 2013  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

"""
Describes and sets up all the properties that are necessary for a generic
genetic algorithm. It then associates those properties with the functions
from the AbstractGA class and creates the base processing structure. This class
is still abstract in the sense that it is designed to be inherited NOT
instantiated. Mainly, no fit function is defined.

Classes:
GenericGA(num, tNum, mProb, cProb, tProb, rProb, elite, k,
          dir_, maxGens) -- generic ga class
GenericGAQ2(num, tNum, mProb, cProb, tProb, rProb, elite, k,
            dir_, maxGens, min_, max_, noise, gSize)
            -- generic ga class with some tweaking for basic mutation use
"""

from AbstractGA import *
from random import random
from random import shuffle

class GenericGA(AbstractGA):
  """Generic GA class setting up properties and basic structure.

  Designed to be inherited by a class that implements a representation for a
  particular GA solvable problem.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens):
    """Sets up all the relevant properties for a particular GA problem.

    Keyword arguments:
    num (int) -- the size of the population
    tNum (int) -- size of the tournaments
    mProb (float) -- probability of mutation
    cProb (float) -- probability of crossover
    tProb (float) -- probability of random tournament selection
    rProb (float) -- selection pressure for rank selection
    elite (bool) -- indicates if keeping best chromosome overall (elitism)
    k (int) -- pool size for heuristic mutation
    dir_ (int) -- max (-1) or min (0) based selection
    maxGens (int) -- max number of generations for one problem run
    
    """

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
    """Template for fitness function.

    Keyword arguments:
    chrom (tuple) -- population member or chromosome

    """
    pass

  def adaptivemodprob(f):
    """Decorator for adaptive mutation and crossover probability adjustment.

    Decreases mutation and crossover probability as the best selected
    chromosome improves.

    Keyword arguments:
    args (tuple) -- holds the object instance, the selection function, the
    mutation function, the crossover function, and the greater or less than
    operator in that order
    kw (tuple) -- just to be generic; isn't used

    """
    def wrapper(*args, **kw):
      try:
        return f(*args, **kw)
      except TypeError:
        self = args[0]
        op = args[-1]
        args = args[:-1]
        avg = sum([self.fitFunction(x) for x in self.pop])/len(self.pop)
        if op(self.best, avg):
          #I did not create this formula. It's from the literature.
          val = (self.max-self.best)/(self.max/avg)
          self.mProb = self.MUT_PROB*val
          self.cProb = self.CROSS_PROB*val
        else:
          self.mProb = self.MUT_PROB
          self.cProb = self.CROSS_PROB
        return f(*args, **kw)
    return wrapper
  
  @adaptivemodprob
  def run(self, selFunction, mutFunction, crossFunction):
    """Basic GA structure; returns current best chromosome and generation.

    This is not a loop so the function has to be called multiple times.
    This allows for clean updating when using threading in the GUI.

    Keyword arguments:
    selFunction (func) -- a selection function
    mutFunction (func) -- a mutation function
    crossFunction (func) -- a crossover function

    """
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
    """Generator that creates never ending pairs of parents.

    Keyword arguments:
    matePool (dict) = chromosomes chosen by selection function to be parents

    """
    half = len(matePool)/2
    while True:
      shuffle(matePool)
      for p1, p2 in zip(matePool[half:], matePool[:half]):
        yield p1, p2

#Binds the relevant properties to each of the functions that need them.
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
  """Generic GA class setting up properties and basic structure.

  Designed to be inherited by a class that implements a representation for a
  particular GA solvable problem.
  Specifically designed for problems that use numeric strings that can
  have repetition and noise fluctuations.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    """Sets up all the relevant properties for a particular GA problem.

    Keyword arguments:
    num (int) -- the size of the population
    tNum (int) -- size of the tournaments
    mProb (float) -- probability of mutation
    cProb (float) -- probability of crossover
    tProb (float) -- probability of random tournament selection
    rProb (float) -- selection pressure for rank selection
    elite (bool) -- indicates if keeping best chromosome overall (elitism)
    k (int) -- pool size for heuristic mutation
    dir_ (int) -- max (-1) or min (0) based selection
    maxGens (int) -- max number of generations for one problem run
    min_ (int) -- min numeric value for chromosome allele
    max_ (int) -- max numeric value for chromosome allele
    noise (int) -- positive have of noise range value
    gSize (int) -- chromosome size/length
    
    """
    super(GenericGAQ2, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                      elite, k, dir_, maxGens)
    
    self.MIN = min_
    self.MAX = max_
    self.CHROM_SIZE = gSize
    self.NOISE = range(-noise, noise+1)
    self.NOISE.remove(0)

    self.pop = super(GenericGAQ2, self).genPop(self.CHROM_SIZE, self.POP_SIZE,
                                               self.MIN, self.MAX)

#Binds relevant properties to basic mutation function

  def basicMutate(self, member):
    return super(GenericGAQ2, self).basicMutate(member, self.mProb, self.NOISE,
                                                self.MIN, self.MAX)
