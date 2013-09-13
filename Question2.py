from GenericGA import GenericGAQ2
import random
from operator import mul
        
class OneMax(GenericGAQ2):

  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    super(OneMax, self).__init__(num, tNum, mProb, cProb, tProb, rProb, elite,
                                 k, dir_, maxGens, min_, max_, noise, gSize)


  def fitFunction(self, chrom):
    return sum(chrom)

class SimpleMax(GenericGAQ2):

  def __init__(self, num, tNum, mProb, cProb, tProb, rProb, elite, k,
               dir_, maxGens, min_, max_, noise, gSize):
    super(SimpleMax, self).__init__(num, tNum, mProb, cProb, tProb, rProb,
                                    elite, k, dir_, maxGens, min_, max_, noise,
                                    gSize)

  def fitFunction(self, chrom):
    return reduce(mul, chrom[:5]) / reduce(mul, chrom[5:])

class LeadingOnes(GenericGAQ2):

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


      
