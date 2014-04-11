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
Describes and implements a collection of generic functions that are used by
genetic algorithms. Despite the inclusion of implementations, the class is
abstract in the sense that it is designed to be inherited NOT instantiated.

Classes:
AbstractGA() -- abstract class of genetic algorithm functions for population
creation, member/chromosome selection, mutation, and crossover
NoEdge() -- exception class that indicates that all the edge options have
been exhausted for a given allele in edge recombination function
"""

from copy import deepcopy
from functools import partial
from itertools import permutations
from random import shuffle
from random import randint
from random import sample
from random import choice
from random import random


class AbstractGA(object):
    """An inheritable class with generic versions of all GA functions learned.

    Designed to be inherited by a class that implements a GA structure and a
    representation for a particular GA solvable problem.

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
        while len(pop) < num:
            pop[tuple(sample(pool, size))] = None
        return pop

    def genPopPermut(self, vals, num):
        """Build random population of tuples through permutation; return dict.

        Keyword arguments:
        vals (list) -- the values to be permuted
        num (int) -- the size of the population
        
        """
        pop = {}
        while len(pop) < num:
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
        """Decorator that iterates the mutation based on probability.

        Checks probability once per allele.
        The trick is that this wrapper takes the probability, uses it, but
        does not pass it along to the mutation function, which doesn't need it.
        It also repackages the new child as a tuple.

        Keyword arguments:
        args (tuple) -- holds the object instance, child to be mutated, and
        mutation probability in that order.
        kw (tuple) -- just to be generic; isn't used.
        
        """
        def wrapper(*args, **kw):
            self, child, prob = args
            for x in child:
                if random() < prob:
                    child = f(self, child)
            return tuple(child)
        return wrapper

    def domultiple2(f):
        """Decorator that iterates the mutation based on probability.

        Checks probability once per allele.
        Designed specifically for the heuristicMutate function.
        The trick is that this wrapper takes the probability, uses it, but
        does not pass it along to the mutation function, which doesn't need it.
        It also repackages the new child as a tuple.

        Keyword arguments:
        args (tuple) -- holds the object instance, K value, fitFunction, sort
        direction, child to be mutated, and mutation probability in that order
        kw (tuple) -- just to be generic; isn't used
        
        """
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
        """Deletes an allele and adds a new one at random position.

        Keyword arguments:
        member (tuple) -- the population member that will be the template for
        mutation

        """
        child = [x for x in member]
        val = choice(child)
        child.remove(val)
        index = randint(0, len(child))
        child[index:index] = [val]
        return tuple(child)

    def mutationBase(self, member):
        """Base function returns child, length of child, and mutation indices.

        Keyword arguments:
        member (tuple) -- the population member that will be the template for
        mutation

        """
        child = [x for x in member]
        ln = len(child)-1
        index1 = randint(0, ln)
        index2 = randint(index1, ln)
        return child, index1, index2

    def inversion(self, args):
        """Mutates by taking a slice and reversing its order.

        Keyword arguments:
        args (tuple) -- holds the child, child length, and mutation indices in
        that order

        """
        child, index1, index2 = args
        slice_ = child[index1:index2]
        slice_.reverse()
        child[index1:index2] = slice_
        return tuple(child)

    @domultiple
    def inversionMutate(self, member):
        """Helper function that joins base function to inversion function.

        Function that is called for inversion mutation.
        Uses the @domultiple decorator for multiple mutation iterations.

        Keyword arguments:
        member (tuple) -- the population member that will be the template for
        mutation

        """
        return self.inversion(self.mutationBase(member))

    def reciprocal(self, args):
        """Swaps two alleles.

        Keyword arguments:
        args (tuple) -- holds the child, child length, and mutation indices in
        that order

        """
        child, index1, index2 = args
        val1 = deepcopy(child[index1])
        val2 = deepcopy(child[index2])
        child[index1] = val2
        child[index2] = val1
        return tuple(child)

    @domultiple
    def reciprocalMutate(self, member):
        """Helper function that joins base function to reciprocal function.

        Function that is called for reciprocal mutation.
        Uses the @domultiple decorator for multiple mutation iterations.

        Keyword arguments:
        member (tuple) -- the population member that will be the template for
        mutation

        """
        return self.reciprocal(self.mutationBase(member))

    @domultiple2
    def heuristicMutate(self, k, fitFunction, dir_, member):
        """Swaps 'k' alleles multiple times and returns the best option.

        Uses the @domultiple2 decorator for multiple mutation iterations.
        Does not use the base mutation function.

        Keyword arguments:
        k (int) -- number of alleles to re-order
        fitFunction -- function that calculates fitness of a member
        dir_ (int) -- max (-1) or min (0) based selection
        member (tuple) -- the population member that will be the template for
        mutation
        """
        children = []
        ln = len(member)
        pool = range(ln)
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
        return tuple(children[fitted[dir_][1]])

##################################################################
#Crossover Functions
##################################################################

    def sortparents(f):
        """Decorator that makes the smaller parent p1 if different sizes.

        Repackages the parents if p1 is larger than p2 then calls the function.

        I don't think this was used because none of the questions required
        different sized parents.

        Keyword arguments:
        args (tuple) -- holds the object instance and the two parents
        kw (tuple) -- just to be generic; isn't used
        
        """
        def wrapper(*args, **kw):
            self, p1, p2 = args
            if len(p1) > len(p2):
                return f(self, p2, p1)
            else:
                return f(*args)
        return wrapper

    def oneCrossover(self, p1, p2):
        """Swaps two sections of p1/2 after a random index; returns 2 children.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        l = len(p1)
        index = randint(0, l)
        c1 = p1[:index]+p2[index:]
        c2 = p2[:index]+p1[index:]
        return c1, c2

    def baseCrossover(self, p1, p2):
        """Base function returns both parents and two indices for crossover.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        l = len(p1)
        index1 = randint(0, l)
        index2 = randint(index1, l)
        return p1, p2, index1, index2

    def addslice(f):
        """Decorate that includes the original parent slices in return values.

        Keyword arguments:
        args (tuple) -- holds object instance, p1, and p2 in that order

        """
        def wrapper(*args, **kw):
            args = f(*args, **kw)
            args = list(args)
            args.extend([args[0][args[2]:args[3]], args[1][args[2]:args[3]]])
            return tuple(args)
        return wrapper

    @addslice
    def baseCrossoverS(self, p1, p2):
        """Helper function that joins base crossover with @addslice decorator.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        return self.baseCrossover(p1, p2)

    def twoPoint(self, args):
        """Swaps the two slices and returns resulting children.

        Keyword arguments:
        args (tuple) -- holds both parents, both indices, and both slices

        """
        p1, p2, index1, index2, slice_1, slice_2 = args
        c1 = p1[:index1]+slice_2+p1[index2:]
        c2 = p2[:index1]+slice_1+p2[index2:]
        return c1, c2

    def twoCrossover(self, p1, p2):
        """Helper function that joins addslice base function to twoPoint swap.

        Function that is called for two point crossover.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        return self.twoPoint(self.baseCrossoverS(p1, p2))

    def partialMap(self, args):
        """Swaps the elements that are not already contained in parent slice.

        This function takes two slices, keeps the alleles that are
        contained in both slices in their original positions, and re-orders the
        rest in a random arrangement in the remaining positions.

        Keyword arguments:
        args (tuple) -- holds both parents, both indices, and both slices

        """
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
        """Helper function that joins addslice base function to partial map.

        Function that is called for partial map crossover.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        return self.partialMap(self.baseCrossoverS(p1, p2))

    def orderMap(self, args, q):
        """Places slice at 'q' after removing overlap between slice an parent.

        This function takes two slices, removes the alleles that are duplicated
        from the parent and slice, and inserts the remaining alleles
        of the slice into the position at the lower index 'q'.

        Keyword arguments:
        args (tuple) -- holds both parents, both indices, and both slices
        q (int) -- the lower index to insert the slice at for both parents

        """
        p1, p2, index1, index2, slice_1, slice_2 = args
        c1 = [x for x in p1 if x not in slice_2]
        c2 = [x for x in p2 if x not in slice_1]
        c1[q:q] = slice_2
        c2[q:q] = slice_1
        return c1, c2

    def orderCrossover(self, p1, p2):
        """Helper function that joins addslice base function to order map.

        Function that is called for order map crossover.
        It also selects 'q' based on the lower index (index1). This is an
        unnecessary step but it is a little cleaner than just using index1 in
        the order map function: it designates a new variable to explicitly
        represent the new role of the index.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        args = self.baseCrossoverS(p1, p2)
        return self.orderMap(args, args[2])

    def injectionMap(self, args):
        """Random ordermap: places slice at random 'q' after removing overlap.

        This function takes two slices, removes the alleles that are duplicated
        from the parent and slice, and inserts the remaining alleles
        of the slice into the position at a random index ('q' in order map).

        Keyword arguments:
        args (tuple) -- holds both parents, both indices, and both slices

        """
        limit = len(args[0]) - len(args[4])
        return self.orderMap(args, randint(0, limit))

    def injectionCrossover(self, p1, p2):
        """Helper function that joins addslice base function to injection map.

        Function that is called for injection map crossover.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        return self.injectionMap(self.baseCrossoverS(p1, p2))

    def positionCrossover(self, p1, p2):
        """Replaces random alleles with unique alleles from other parent.

        The alleles from the other parent are placed in the original positions
        in the reverse order that they occurred.
        This function does not need the base crossover or slicing decorator.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
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
#
#Uses the edges matrix data structure which is a dictionary (or hash
#table) with the alleles as the keys and a list of two lists as the
#associated value. The first internal list has the alleles that are
#connected in both parents to the key allele and the second list has
#alleles that are connected to the key in one parent only.
#The two ends of the chromosome are assumed to connect.
#
#E.g.,
#chromosome1 : 12345
#chromosome2 : 23514
#{3 : [[2], [5, 4]],
# 1 : [[5], [4, 2]], ...}

    def edgeLoop(self, gen, edges):
        """Populates the edge matrix data structure.

        For each pair of alleles (i.e., edge) in each of the chromosomes, this
        proceeds by trying to remove one of the alleles (1) from the second
        internal list of the other allele (2). If successful, this indicates
        that the edge (1-2) is shared and the allele (1) is added to the first
        internal list. If the removal fails, it indicates it's the first
        occurrence of that edge so the allele (1) is appended to the second
        internal list. This repeats with the other allele (1) in the pair as
        the key.
        edges is edited in place (i.e., by reference) for speed.

        Keyword arguments:
        gen (generator) -- a generator of 2-element window through chromosome
        edges (dict) -- a dictionary of the edges data structure

        """
        n = list(gen)
        for pair in n:
            e = edges[pair[0]]
            #This isn't elegant but it works and it's fast.
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

        Keyword arguments:
        chromosome (tuple) -- the chromosome to be windowed

        """
        index1 = -1
        index2 = 0
        while index2 < len(chromosome):
            yield chromosome[index1], chromosome[index2]
            index1 += 1
            index2 += 1

    def buildEdgeMatrix(self, p1, p2):
        """Builds the complete edge matrix data structure and returns it.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
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
        """Removes elem from all internal lists in the curEdges edge matrix.

        This function works by finding all the values that were connected to
        the elem allele, going through each of them in the edge matrix, and
        removing the elem allele from the internal list.
        Since the presence of the allele in the second internal list is more
        likely, we search in the first internal list only after we fail to
        remove the allele from the second.
        curEdges is edited in place (i.e., by reference) for speed.

        Keyword arguments:
        edges (dict) -- the original, unedited edge matrix
        curEdges (dict) -- the current state of the edited edge matrix
        elem (int) -- the allele that needs to be removed

        """
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
        """Gets the next allele from the curElem's connecting alleles.

        The next alleles are picked from the shared connecting alleles (first
        interna list) first, then from the allele in the second list with the
        fewest connections in their second internal list.
        If there are still no options then it raises the NoEdge Exception. We
        are either done or we have hit a dead end.

        Keyword arguments:
        edges (dict) -- the edge matrix data structure
        curElem (int) -- the current element

        """
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
        """Performs the main edge recombination function and returns new child.

        This builds a new child allele by allele using all the previous
        functions and starting with a random allele. If it runs out of
        connecting alleles, it checks if the child is complete. If it is, it
        returns it. If it's not, then it reverses the order of the child and
        continues from the new end allele. If it still has no option, it tries
        a new random start allele. Once all start alleles are exhausted, it
        returns None.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
        edges = self.buildEdgeMatrix(p1, p2)
        startsLeft = edges.keys()
        ln = len(startsLeft)
        shuffle(startsLeft)
        for val in startsLeft:
            curEdges = self.copier(edges)
            c1 = [val]
            self.removeEdges(edges, curEdges, c1[-1])
            while True:
                try:
                    elem = self.getEdge(curEdges, c1[-1])
                except NoEdge:
                    if len(c1) == ln:
                        return c1
                    c1.reverse()
                    try:
                        elem = self.getEdge(curEdges, c1[-1])
                    except NoEdge:
                        break
                c1.append(elem)
                self.removeEdges(edges, curEdges, elem)
        return None

    def copier(self, edges):
        """Performs a deep copy of the dictionary of edges.

        Keyword arguments:
        edges (dict) -- the edges data structure

        """
        edges_ = {}
        for k in edges:
            edges_[k] = [[x for x in edges[k][0]], [x for x in edges[k][1]]]
        return edges_

    def edgeRecombUI(self, p1, p2):
        """Performs the edgeRecomb function once for each child.

        If no new combination can be found for a given child, it keeps the
        parent instead.

        Keyword arguments:
        p1 (tuple) -- first parent
        p2 (tuple) -- second parent

        """
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
        """Base function returns both parents and two indices for crossover.

        By using a dictionary as the data structure for the population it is
        easy to get a new batch of unique chromosomes of the previous pop size:
        identical chromosomes overwrite the same hash value in the dictionary.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        typeFitFunction (func) -- partial function that scores fitness
        typeSearch (func)  -- partial function for finding next chromosome by
        using the scores
        dir_ (int) -- max (-1) or min (0) based selection
        elite (bool) -- indicates if keeping best chromosome overall (elitism)

        """
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

    def tournamentFit(self, pop, fitFunction):
        """Scores and sorts the chromosomes for future tournaments.

        Helper function for tournament selection.
        One of the typeFitFunctions.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness
        elite (bool) -- indicates if keeping best chromosome overall (elitism)

        """
        fitProb = [[fitFunction(parent), pos] for pos, parent in \
                   enumerate(pop)]
        fitProb.sort()
        return fitProb        

    def tournamentSel(self, pop, fitFunction, size, prob, dir_, elite=False):
        """Prepares all the functions for base selection function.

        Partial functions are used to simplify the variable passing between
        functions.
        This is another helper function that stitches the relevant parts
        together.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness
        size (int) -- size of the tournament
        prob (float) -- probability that tournament winner is picked randomly
        dir_ (int) -- max (-1) or min (0) based selection
        elite (bool) -- indicates if keeping best chromosome overall (elitism)

        """
        return self.baseSel(pop, partial(self.tournamentFit, \
                                         fitFunction=fitFunction), \
                            partial(self.tourney, size=size, prob=prob,
                                    dir_=dir_),
                            dir_, elite)

    def tourney(self, fitProb, size, prob, dir_):
        """Prepares all the functions for base selection function.

        Partial functions are used to simplify the variable passing between
        functions.
        This is another helper function that stitches the relevant parts
        together.

        Keyword arguments:
        fitProb (list) -- scored and sorted list of chromosomes
        size (int) -- size of the tournament
        prob (float) -- probability that tournament winner is picked randomly
        dir_ (int) -- max (-1) or min (0) based selection

        """
        pool = sample(fitProb, size)
        pool.sort()
        if random() < prob:
            return pool[dir_][1]
        else:
            return choice(pool[:-1])[1]

    def rouletteFit(self, pop, fitFunction):
        """Scores and sorts the chromosomes for roulette selection.

        Helper function for roulette selection.
        One of the typeFitFunctions.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness

        """
        fitProb = [fitFunction(parent) for parent in pop]
        popFit = float(sum(fitProb))
        fitProb = [[parentFit/popFit, pos] for pos, parentFit in \
                   enumerate(fitProb)]
        fitProb.sort()
        fitProb = self.calcCDF(fitProb)
        return fitProb

    def rouletteSel(self, pop, fitFunction, dir_, elite=False):
        """Prepares all the functions for base selection function.

        Partial functions are used to simplify the variable passing between
        functions.
        This is another helper function that stitches the relevant parts
        together.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness
        dir_ (int) -- max (-1) or min (0) based selection
        elite (bool) -- indicates if keeping best chromosome overall (elitism)

        """
        return self.baseSel(pop, partial(self.rouletteFit, \
                                         fitFunction=fitFunction),
                            self.binSearch, dir_, elite)

    def rankFit(self, pop, fitFunction, rankFunction):
        """Scores and sorts the chromosomes for rank selection then returns CDF.

        Helper function for rank selection.
        One of the typeFitFunctions.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness

        """
        fitProb = [[fitFunction(parent), pos] for pos, parent in \
                   enumerate(pop)]
        ln = len(fitProb)
        fitProb.sort()
        fitProb = [[rankFunction(pos, ln), x[1]] for pos, x in
                   enumerate(fitProb)]
        fitProb = self.calcCDF(fitProb)
        return fitProb

    def rankSel(self, pop, fitFunction, rankFunction, selPressure, dir_,
                elite=False):
        """Prepares all the functions for base selection function.

        Partial functions are used to simplify the variable passing between
        functions.
        This is another helper function that stitches the relevant parts
        together.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes
        fitFunction (func) -- function that scores fitness
        rankFunction (func) -- for different ranking formulas
        selPressure (float) -- varies between 1 and 2 with lower numbers
        making selection easier
        dir_ (int) -- max (-1) or min (0) based selection
        elite (bool) -- indicates if keeping best chromosome overall (elitism)

        """
        return self.baseSel(pop, partial(self.rankFit, fitFunction=fitFunction,
                                         rankFunction=partial(rankFunction,
                                                              sp=selPressure)),
                            self.binSearch, dir_, elite)

    def linearRank(self, rank, num, sp):
        """Calculates the linear rank chromosomes based on selection pressure.

        Keyword arguments:
        rank (int) -- the current rank of the chromosome with higher == better
        num (int) -- current number of chromosomes in population
        sp (float) -- selection pressure; varies between 1 and 2 with lower
        numbers making selection easier

        """
        return ((2-sp)/num) + 2*rank/(num*(num-1))

    def calcCDF(self, pop):
        """Builds a cumulative distribution function based on roulette scoring.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes

        """
        pop_ = deepcopy(pop)
        total = 0
        for pos, pair in enumerate(pop_):
            total += pair[0]
            pop_[pos][0] = total
        return pop_

    def binSearch(self, pop):
        """Binary search to find rank or roulette bin for given probability.

        Keyword arguments:
        pop (dict) -- a dictionary with keys as population of all chromosomes

        """
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
        """Builds a pool of probabilities for faster selection.

        Not used.

        Keyword arguments:
        size (int) -- size of the population
        prob (float) -- desired probability
        """
        probPool = [x for x in range(size) for y in range(int(1000*prob))]
        probPool.extend([None for x in range(size*int(1000*(1-prob)))])
        return probPool
            
class NoEdge(Exception):
    """Exception class for edge recombination functions."""
    pass

