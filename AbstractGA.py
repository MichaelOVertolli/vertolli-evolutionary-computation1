from copy import deepcopy
from functools import partial
from itertools import permutations
from random import shuffle
from random import randint
from random import sample
from random import choice
from random import random
import gc


class AbstractGA(object):
    """An inheritable class with generic versions of all GA functions learned.

    Designed to be inherited by a class that implements a representation for a
    particular GA solvable problem.

    Public Methods:
    genPop(size, num, max_, min_)
    genPopPermut(vals, num)

    rouletteSel(pop, fitFunction, dir_, elite)
    tournamentSel(pop, fitFunction, size, prob, dir_, elite)

    basicMutate(member, prob, noise, min_, max_)
    insertionMutate(member, prob)
    inversionMutate(member, prob)
    reciprocalMutate(member, prob)
    heuristicMutate(k, fitFunction, member, dir_, prob)

    oneCrossover(p1, p2)
    twoCrossover(p1, p2)
    partialCrossover(p1, p2)
    orderCrossover(p1, p2)
    injectionCrossover(p1, p2)
    positionCrossover(p1, p2)
    edgeRecombUI(p1, p2)
    
    """

    def __init__(self):
        pass

    def genPop(self, size, num, min_, max_):
        """Build random population of tuples and return as keys in dictionary.

        Keyword arguments:
        size (int) -- the size of tuple
        num (int) -- the size of the population
        min_ (int) -- the minimum value a tuple cell can hold 
        max_ (int) -- the maximum value a tuple cell can hold 
        
        """
        pop = {}
        #The +1 accommodates for the 0 index, which allows max_ to be generic.
        pool = [x for x in range(min_, max_+1) for y in range(size)]
        while len(pop.keys()) < num:
            pop[tuple(sample(pool, size))] = None
        return pop

    def genPopPermut(self, vals, num):
        #I'm getting tired of refactoring. I'll refactor later.
        #Use generators to refactor.
        pop = {}
        while len(pop.keys()) < num:
            shuffle(vals)
            pop[tuple(vals)] = None
        return pop

##################################################################
#Mutation Functions
##################################################################

    def basicMutate(self, member, prob, noise, min_, max_):
        """Basic stochastic mutation of the parent returning child tuples.

        Keyword arguments:
        member (tuple) -- the population member that will be the template for
        mutation
        prob (float) --  the probability of a mutation occurring at an allele
        noise (list) -- an array of values for degree of mutation; should not
        include 0
        min_ (int) -- the minimum value a tuple cell can hold 
        max_ (int) -- the maximum value a tuple cell can hold; must be less
        than 1,000,000

        """
        #It didn't seem worth making a separate function to encapsulate this
        #one line of code, even if it is duplicated.
        child = [x for x in member]
        child = [x if random() > prob else x+choice(noise) for x in child]
        child = [x if x <= max_ else max_ for x in child]
        child = [x if x >= min_ else min_ for x in child]
        return tuple(child)

    def domultiple(f):
        def wrapper(*args, **kw):
            inst, child, prob = args
            for x in child:
                if random() < prob:
                    child = f(inst, child)
            return tuple(child)
        return wrapper

    def domultiple2(f):
        def wrapper(*args, **kw):
            prob = args[-1]
            child = args[-2]
            for x in child:
                if random() < prob:
                    child = f(*args[:-1])
            return tuple(child)
        return wrapper

    @domultiple
    def insertionMutate(self, member):
        child = [x for x in member]
        val = choice(child)
        child.remove(val)
        index = randint(0, len(child))
        child[index:index] = [val]
        return tuple(child)

    def mutationBase(self, member):
        child = [x for x in member]
        l = len(child)-1
        index1 = randint(0, l)
        index2 = randint(index1, l)
        return child, l, index1, index2

    def inversion(self, args):
        child, l, index1, index2 = args
        slice_ = child[index1:index2]
        slice_.reverse()
        child[index1:index2] = slice_
        return tuple(child)

    @domultiple
    def inversionMutate(self, member):
        return self.inversion(self.mutationBase(member))

    def reciprocal(self, args):
        child, l, index1, index2 = args
        val1 = deepcopy(child[index1])
        val2 = deepcopy(child[index2])
        child[index1] = val2
        child[index2] = val1
        return tuple(child)

    @domultiple
    def reciprocalMutate(self, member):
        return self.reciprocal(self.mutationBase(member))

    @domultiple2
    def heuristicMutate(self, k, fitFunction, dir_, member):
        children = []
        l = len(member)
        pool = range(l)
        shuffle(pool)
        indices = sample(pool, k)
        alleles = [member[x] for x in indices]
        alleles = list(permutations(alleles, k))
        for set_ in alleles:
            child = [x for x in member]
            for pos, index in enumerate(indices):
                child[index] = set_[pos]
            children.append(child)
        fitted = [(fitFunction(child), pos) for pos, child in \
                  enumerate(children)]
        fitted.sort()
        return tuple(children[fitted[-1][1]])

##################################################################
#Crossover Functions
##################################################################

    def sortparents(f):
        """Makes sure the smaller parent is p1 in case of different sizes."""
        def wrapper(*args, **kw):
            self, p1, p2 = args
            if len(p1) > len(p2):
                return f(self, p2, p1)
            else:
                return f(*args)
        return wrapper

    def oneCrossover(self, p1, p2):
        l = len(p1)
        index = randint(0, l)
        c1 = p1[:index]+p2[index:]
        c2 = p2[:index]+p1[index:]
        return c1, c2

    def baseCrossover(self, p1, p2):
        l = len(p1)
        index1 = randint(0, l)
        index2 = randint(index1, l)
        return p1, p2, index1, index2

    def addslice(f):
        """Adds slices of parents to return values."""
        def wrapper(*args, **kw):
            args = f(*args, **kw)
            args = list(args)
            args.extend([args[0][args[2]:args[3]], args[1][args[2]:args[3]]])
            return tuple(args)
        return wrapper

    @addslice
    def baseCrossoverS(self, p1, p2):
        return self.baseCrossover(p1, p2)

    def twoPoint(self, args):
        p1, p2, index1, index2, slice_1, slice_2 = args
        c1 = p1[:index1]+slice_2+p1[index2:]
        c2 = p2[:index1]+slice_1+p2[index2:]
        return c1, c2

    def twoCrossover(self, p1, p2):
        return self.twoPoint(self.baseCrossoverS(p1, p2))

    def partialMap(self, args):
        p1, p2, index1, index2, slice_1, slice_2 = args
        nc1 = list(set(slice_1) - set(slice_2))
        shuffle(nc1)
        nc1P = nc1.pop
        nc2 = list(set(slice_2) - set(slice_1))
        shuffle(nc2)
        nc2P = nc2.pop
        c1 = [x if (pos >= index1 and pos < index2) or x not in slice_2 else \
              nc1P() for pos, x in enumerate(p1)]
        c2 = [x if (pos >= index1 and pos < index2) or x not in slice_1 else \
              nc2P() for pos, x in enumerate(p2)]
        c1 = c1[:index1]+slice_2+c1[index2:]
        c2 = c2[:index1]+slice_1+c2[index2:]
        return c1, c2

    def partialCrossover(self, p1, p2):
        return self.partialMap(self.baseCrossoverS(p1, p2))

    def orderMap(self, args, q):
        p1, p2, index1, index2, slice_1, slice_2 = args
        c1 = [x for x in p1 if x not in slice_2]
        c2 = [x for x in p2 if x not in slice_1]
        c1[q:q] = slice_2
        c2[q:q] = slice_1
        return c1, c2

    def orderCrossover(self, p1, p2):
        args = self.baseCrossoverS(p1, p2)
        return self.orderMap(args, args[2])

    def injectionMap(self, args):
        limit = len(args[0]) - len(args[4])
        return self.orderMap(args, randint(0, limit))

    def injectionCrossover(self, p1, p2):
        return self.injectionMap(self.baseCrossoverS(p1, p2))

    def positionCrossover(self, p1, p2):
        c1 = [x if randint(0,1) == 1 else None for x in p1]
        c2 = [x if randint(0,1) == 1 else None for x in p2]
        nc1 = [x for x in p2 if x not in c1]
        nc1.reverse()
        nc1P = nc1.pop
        nc2 = [x for x in p1 if x not in c2]
        nc2.reverse()
        nc2P = nc2.pop
        c1 = [x if x != None else nc1P() for x in c1]
        c2 = [x if x != None else nc2P() for x in c2]
        return c1, c2

##################################################################
#Edge Recombination Function
##################################################################

    def edgeLoop(self, gen, edges):
        n = list(gen)
        for pair in n:
            e = edges[pair[0]]
            #This isn't as elegant but it works and it's fast.
            try:
                e[1].remove(pair[1])
            except ValueError:
                e[1].append(pair[1])
            else:
                e[0].append(pair[1])
            e = edges[pair[1]]
            try:
                e[1].remove(pair[0])
            except ValueError:
                e[1].append(pair[0])
            else:
                e[0].append(pair[0])

    def nextTwo(self, chromosome):
        """Generator function that yields 2-element window through chromosome.

        This currently includes the ends as a valid window so the chromosome
        is like a string with connected ends (i.e., circle).

        """
        index1 = -1
        index2 = 0
        while index2 < len(chromosome):
            yield chromosome[index1], chromosome[index2]
            index1 += 1
            index2 += 1

    def buildEdgeMatrix(self, p1, p2):
        try:
            edges = self.copier(self.edges)
        except AttributeError:
            #The first internal list are edges that are shared between
            #parents. The second internal list are standard edges.
            edges = dict((elem, [[], []]) for elem in p1)
            #I'm really trying to squeeze out speed. This assignment
            #prevents repeated creation of the empty matrix.
            self.edges = self.copier(edges)
        p1Gen = self.nextTwo(p1)
        p2Gen = self.nextTwo(p2)
        self.edgeLoop(p1Gen, edges)
        self.edgeLoop(p2Gen, edges)
        return edges

    def removeEdges(self, edges, curEdges, elem):
        vals = edges[elem][0]+edges[elem][1]
        for k in vals:
            k = curEdges[k]
            try:
                k[1].remove(elem)
            except ValueError:
                try:
                    k[0].remove(elem)
                except ValueError:
                    pass

    def getEdge(self, edges, curElem):
        mutual = edges[curElem][0]
        mono = edges[curElem][1]
        try:
            elem = mutual.pop()
        except IndexError:
            mono.sort(key=lambda val: len(edges[val][1]))
            try:
                return mono[0]
            except IndexError:
                raise NoEdge()
        else:
            return elem
                

    def edgeRecomb(self, p1, p2):
        edges = self.buildEdgeMatrix(p1, p2)
        startsLeft = edges.keys()
        l = len(startsLeft)
        shuffle(startsLeft)
        for val in startsLeft:
            curEdges = self.copier(edges)
            c1 = [val]
            self.removeEdges(edges, curEdges, c1[-1])
            while True:
                try:
                    elem = self.getEdge(curEdges, c1[-1])
                except NoEdge:
                    if len(c1) == l:
                        return c1
                    c1.reverse()
                    try:
                        elem = self.getEdge(curEdges, c1[-1])
                    except NoEdge:
                        break
                c1.append(elem)
                self.removeEdges(edges, curEdges, elem)

    def copier(self, edges):
        edges_ = {}
        for k in edges.keys():
            edges_[k] = [[x for x in edges[k][0]], [x for x in edges[k][1]]]
        return edges_

    def edgeRecombUI(self, p1, p2):
        c1 = self.edgeRecomb(p1, p2)
        if c1 == None:
            c1 = p1
        c2 = self.edgeRecomb(p1, p2)
        if c2 == None:
            c2 = p2
        return c1, c2

        
##################################################################
#Selection Functions
##################################################################

    def baseSel(self, pop, typeFitFunction, \
                typeSearch, dir_, elite=False):
        n = len(pop)
        matePool = []
        fitProb = typeFitFunction(pop)
        f = pop[fitProb[dir_][1]]
        if elite:
            matePool.append(f)
        while len(matePool) < n:
            index = typeSearch(fitProb)
            matePool.append(pop[int(index)])
        return matePool, f

    def tournamentFit(self, pop, fitFunction, elite):
        fitProb = [[fitFunction(parent), pos] for pos, parent in \
                   enumerate(pop)]
        fitProb.sort()
        return fitProb        

    def tournamentSel(self, pop, fitFunction, size, prob, dir_, elite=False):
        return self.baseSel(pop, partial(self.tournamentFit, \
                                         fitFunction=fitFunction,
                                         elite=elite), \
                            partial(self.tourney, size=size, prob=prob,
                                    dir_=dir_),
                            dir_, elite)

    def tourney(self, fitProb, size, prob, dir_):
        pool = sample(fitProb, size)
        pool.sort()#key=lambda fitPair: fitPair[0])
        if random() < prob:
            return pool[dir_][1]
        else:
            return choice(pool[:-1])[1]

    def rouletteFit(self, pop, fitFunction):
        fitProb = [fitFunction(parent) for parent in pop]
        popFit = float(sum(fitProb))
        fitProb = [[parentFit/popFit, pos] for pos, parentFit in \
                   enumerate(fitProb)]
        fitProb.sort()
        fitProb = self.calcCDF(fitProb)
        return fitProb

    def rouletteSel(self, pop, fitFunction, dir_, elite=False):
        return self.baseSel(pop, partial(self.rouletteFit, \
                                         fitFunction=fitFunction),
                            self.binSearch, dir_, elite)

    def rankFit(self, pop, fitFunction, rankFunction):
        fitProb = [[fitFunction(parent), pos] for pos, parent in \
                   enumerate(pop)]
        l = len(fitProb)
        fitProb.sort()
        fitProb = [[rankFunction(pos, l), x[1]] for pos, x in
                   enumerate(fitProb)]
        fitProb = self.calcCDF(fitProb)
        return fitProb

    def rankSel(self, pop, fitFunction, rankFunction, selPressure, dir_,
                elite=False):
        return self.baseSel(pop, partial(self.rankFit, fitFunction=fitFunction,
                                         rankFunction=partial(rankFunction,
                                                              sp=selPressure)),
                            self.binSearch, dir_, elite)

    def linearRank(self, rank, num, sp):
        return ((2-sp)/num) + 2*rank/(num*(num-1))

    def calcCDF(self, pop):
        pop_ = deepcopy(pop)
        total = 0
        for pos, pair in enumerate(pop_):
            total += pair[0]
            pop_[pos][0] = total
        return pop_

    def binSearch(self, pop):
        val = random()
        subPop = pop
        index = 1
        while index > 0:
            index = len(subPop)/2
            if val < subPop[index][0]:
                if val > subPop[index-1][0]:
                    return subPop[index][1]
                else:
                    subPop = subPop[:index]
            else:
                subPop = subPop[index:]
        return 0

    def buildProbPool(self, size, prob):
        probPool = [x for x in range(size) for y in range(int(1000*prob))]
        probPool.extend([None for x in range(size*int(1000*(1-prob)))])
        return probPool
            
class NoEdge(Exception):
    pass

