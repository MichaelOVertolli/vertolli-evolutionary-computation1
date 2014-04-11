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
Various simple problems to test GA implementation.

Classes:
OneMax(num, tNum, mProb, cProb, tProb, rProb, elite, k,
       dir_, maxGens, min_, max_, noise, gSize)
       -- class that implements max binary string problem
SimpleMax(num, tNum, mProb, cProb, tProb, rProb, elite, k,
          dir_, maxGens, min_, max_, noise, gSize)
          -- class that implements multiply half and divide half problem
LeadingOnes(num, tNum, mProb, cProb, tProb, rProb, elite, k,
            dir_, maxGens, min_, max_, noise, gSize)
            -- class that implements sum leading ones problem

Implementations are consistent with GenericGAQ2 superclass. I've included
keyword arguments for the first sub-class for convenience.
"""

from GenericGA import GenericGAQ2
import random
from operator import mul
        
class OneMax(GenericGAQ2):
  """Evolves the max scoring binary string from a random start using a GA.

  fitFunction is a simple sum of all 1's in the string

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
    super(OneMax, self).__init__(num, tNum, mProb, cProb, tProb, rProb, elite,
                                 k, dir_, maxGens, min_, max_, noise, gSize)


  def fitFunction(self, chrom):
    return sum(chrom)

class SimpleMax(GenericGAQ2):
  """Evolves the max scoring binary string from a random start using a GA.

  fitFunction multiplies first half and second half then divides the first half
  by the second half.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    super(SimpleMax, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                    elite, k, dir_, maxGens, min_, max_, noise,
                                    gSize)

  def fitFunction(self, chrom):
    return reduce(mul, chrom[:5]) / reduce(mul, chrom[5:])

class LeadingOnes(GenericGAQ2):
  """Evolves the max scoring binary string from a random start using a GA.

  fitFunction sums all the ones to the first zero in the binary string.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    super(LeadingOnes, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                      elite, k, dir_, maxGens, min_, max_,
                                      noise, gSize)

  def fitFunction(self, chrom):
    try:
      temp = chrom[:chrom.index(0)]
    except ValueError:
      temp = chrom
    return sum(temp)


      
