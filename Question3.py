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
Travelling salesman problem for a GA.

Classes:
TSPGA(num, tNum, mProb, cProb, tProb, rProb, elite, k,
      dir_, maxGens, file_) -- Implements a basic travelling salesman problem
TSPGA1() -- convenience class for testing
"""

from GenericGA import GenericGA
from math import sqrt

class TSPGA(GenericGA):
  """Evolves the near best solution to two travelling salesman problems.

  fitFunction calculates the Euclidean distance of the path between all points.

  Public Methods:
  run(self, selFunction, mutFunction, crossFunction)

  """
  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, file_):
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
    file (string) -- name of the file with the TSP representation
    
    """
    super(TSPGA, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                elite, k, dir_, maxGens)

    self.MAPPED = self.load(file_)

    self.pop = self.genPopPermut(self.MAPPED.keys(), self.POP_SIZE)

  def load(self, file_):
    """Loads the city positions for the TSP into a dictionary that is returned.

    The format in the file is:
    City#(int) X(float) Y(float)

    Keyword arguments:
    file (string) -- name of the file with the TSP representation
    
    """
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
    """Calculates the Euclidean distance for the path travelled.

    Keyword arguments:
    chrom (tuple) -- popultion member or chromosome representing a path

    """
    gen = self.nextTwo(chrom)
    total = 0
    for pair in list(gen):
      total += self.euclidian(self.MAPPED[pair[0]], self.MAPPED[pair[1]])
    return total

  def euclidian(self, point1, point2):
    """Euclidan distance between two points.

    Keyword arguments:
    point1 (list) -- a list of x, y float coordinates city
    point2 (list) -- a list of x, y float coordinates for next city

    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

  def prep(self, chrom, divider):
    """Prepares the current best chromosome to be drawn and scales it to window.

    Keyword arguments:
    chrom (tuple) -- popultion member or chromosome representing a path
    divider (int) -- scaling factor

    """
    gen = self.nextTwo(chrom)
    gen = list(gen)
    gen = [(self.MAPPED[pair[0]], self.MAPPED[pair[1]]) for pair in gen]
    gen = [[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] for pair in gen]
    for pos, _ in enumerate(gen):
      for pos2, __ in enumerate(_):
        gen[pos][pos2] = gen[pos][pos2]/divider
    return gen
    
class TSPGA1(TSPGA):
  def __init__(self):
    super(TSPGA1, self).__init__(500, 10, 0.001, 0.7, 0.6, False, 3, 0,
                                 2000, 'berlin52.txt')
  def go(self):
    return self.run(self.tournamentSel, self.heuristicMutate,
                    self.edgeRecombUI)

