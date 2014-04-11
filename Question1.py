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
A string search problem with GA setup for solution.

Classes:
StrSearch(num, tNum, mProb, cProb, tProb, rProb, elite, k,
          dir_, maxGens, min_, max_, noise, gSize, query)
          -- class that implements a string search problem
"""

import random
import string
import copy
from GenericGA import GenericGAQ2

class StrSearch(GenericGAQ2):
  """Evolves the correct string from a random string using a GA.

  Only lowercase letters and spaces are valid.
  At least 30 characters must be entered.
  Only uses mutation despite inclusion of properties for crossover.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize, query):
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
    query (string) -- the string to search for/evolve
    
    """
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
    """Decorator for adaptive mutation and crossover probability adjustment.

    Decreases mutation and crossover probability as the best selected
    chromosome improves.
    Re-implemented for easy use on the new run function.

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
        avg = sum([self.fitFunction(x) for x in self.pop.keys()])/len(self.pop)
        if op(self.best, avg):
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
    """String search structure; returns best chromosome and generation.

    Overwrites generic run structure for this specific problem's structure.

    Keyword arguments:
    selFunction (func) -- a selection function
    mutFunction (func) -- a mutation function
    crossFunction (func) -- a crossover function

    """
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
    """Determines relative distance of each letter in chromsome and returns score.

    Convenience function for printing the scores to the screen.

    Keyword arguments:
    chrom (tuple) -- a tuple of the numbers representing each character in the
    string

    """
    try:
      return sum([abs(x - y) for x, y in zip(chrom, self.query)])
    except TypeError:
      return sum([abs(x - y) for x, y in zip(self.convert(chrom), self.query)])

  def getClosest(self):
    """Real fit function."""
    bVal = 1000000
    bestStr = []
    for obj in self.pop:
      val = sum([abs(x - y) for x, y in zip(obj, self.query)])
      if val < bVal:
        bVal = val
        bestStr = obj
    return bestStr, bVal

  def switchRep(self, val):
    """Switches a character to a number or vice versa.

    Number to character switch is more common so faster to do it first.

    Keyword arguments:
    val (char or int) -- the current value to switch representation for

    """
    try:
      return self.NUM_TO_CHAR[val]
    except TypeError:
      return self.CHAR_TO_NUM[val]

  def convert(self, obj):
    """Converts an entire string to tuple of ints or vice versa.

    Keyword arguments:
    obj (string or tuple) -- the object to convert

    """
    newStr = ''
    if type(obj) == tuple:
      newStr = ''.join(map(self.switchRep, list(obj)))
    else:
      newStr = map(self.switchRep, list(obj))
    return newStr
