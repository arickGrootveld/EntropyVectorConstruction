## Functions to calculate the entropy vectors of pure and mixed states
import numpy as np
import itertools
import math
import copy
from utilities import partialTrace_TargBits, vnEntropy, qudit, h, newtonsMethod_EntropyFunctional, purifyMixedState

from fpylll import IntegerMatrix, LLL, CVP

class entropyVector_MixedState:
    """
    Entropy vector class to make things easier to work with 
    (specifically to make labelling the elements of the entropy vector
    easier)
    """

    def __init__(self, densityMat, bitPartitions):

        '''
        :param densityMat: Density Matrix corresponding to the state
        :param partitionBits: List of lists that specify which bits belong to which partitions
        '''
        # Checking that the dimension of the density matrix 
        # will match the number of bits specified
        numBits = np.sum([len(targList) for targList in bitPartitions])
        if not (np.log2(densityMat.shape[0]) == numBits):
            
            raise Exception("Number of bits was {}, and the size of the density matrix was {}, " \
                            "which is incompatible".format(numBits, densityMat.shape[0]))
        
        entVec, subSystemCombs = compEntropyVec_MixedState(densityOp=densityMat, bitPartitions=bitPartitions, returnLabels=True)

        self.densityMatrix = densityMat
        self.subSystemLabels = bitPartitions
        self.entVec = entVec
        self.entVecLabels = subSystemCombs

class entropyVector_PureState:
    """
    Attributes
    ----------
    pureState : np.ndarray
        High dimensional pure state vector
    subSystemLabels : str
        Names for each of the subsystems
    entVec : np.ndarray
        Vector of entropy values for the subsystems
    entVecLabels: list[int]
        Combinations of sub-system labels corresponding to each entropy
    """
    def __init__(self, pureState, bitPartitions):
        """
        :param pureState: Vector representing the pure state
        :param partitionBits: List of lists that specify which bits belong to which partitions
        """
        
        # Checking that the dimension of the pure state 
        # matches the number of bits specified
        numBits = np.sum([len(targList) for targList in bitPartitions])
        if not (np.log2(pureState.shape[0]) == numBits):
            raise Exception("Number of bits was {}, and the size of the density matrix was {}, " \
                            "which is incompatible".format(numBits, pureState.shape[0]))
        
        # Normalizing the pure state (just in case)
        pureState = pureState / np.linalg.norm(pureState)

        entVec, subSystemCombs = compEntropyVec_PureState(pureState=pureState, bitPartitions=bitPartitions, returnLabels=True)
    
        self.pureState = pureState
        self.subSystemLabels = bitPartitions
        self.entVec = entVec
        self.entVecLabels = subSystemCombs

    
    

    def __add__(self, other):
        if not isinstance(other, entropyVector_PureState):
            raise Exception("Cannot add pure state entropy vector and {}".format(other))
        
        (combState, combLabels) = entVecAdd_PureState(self, other)
        
        return(entropyVector_PureState(pureState=combState, bitPartitions=combLabels))
            
    
    def __mul__(self, other):
        if not (isinstance(other, int)):
            raise Exception("Cannot multiply a pure state entropy vector and {}".format(other))
        if not (other > 0):
            raise Exception("Cannot multiply an entropy vector by {} (a non-positive integer)".format(other))
        
        (scaledState, scaledLabels) = entVecScalarMult_PureState(entVec=self, scalar=other)        
        return(entropyVector_PureState(pureState=scaledState, bitPartitions=scaledLabels))

    def __rmul__(self, other):
        if not (isinstance(other, int)):
            raise Exception("Cannot multiply a pure state entropy vector and {}".format(other))
        if not (other > 0):
            raise Exception("Cannot multiply an entropy vector by {} (a non-positive integer)".format(other))
        
        (scaledState, scaledLabels) = entVecScalarMult_PureState(entVec=self, scalar=other)        
        return(entropyVector_PureState(pureState=scaledState, bitPartitions=scaledLabels))

 
def entVecAdd_PureState_Reduc(entVec1_Store: tuple[list, np.ndarray], entVec2_Store: tuple[list, np.ndarray]):
    """
    A modified version of the below function that doesn't require all the properties
    of entropy vectors be present, i.e. we don't have to re-compute the entropy vector
    each time we add something
    
    :param entVec1_Store: A tuple with the first element being the labels of an entropy vector, and the second being the pure state
    :param entVec2_Store: A tuple with the first element being the labels of an entropy vector, and the second being the pure state
    """
    labelArray1 = bitmap2Array(entVec1_Store[0])
    labelArray2 = bitmap2Array(entVec2_Store[0])
    combinedLabels = array2Bitmap(labelArray1 + labelArray2)

    combinedPureState = np.kron(entVec1_Store[1], entVec2_Store[1])

    return(combinedPureState, combinedLabels)

def entVecAdd_PureState(entVec1: entropyVector_PureState, entVec2: entropyVector_PureState):
    labelArray1 = bitmap2Array(entVec1.subSystemLabels)
    labelArray2 = bitmap2Array(entVec2.subSystemLabels)
    combinedLabels = array2Bitmap(labelArray1 + labelArray2)

    combinedPureState = np.kron(entVec1.pureState, entVec2.pureState)
    
    return(combinedPureState, combinedLabels)

def entVecScalarMult_PureState(entVec: entropyVector_PureState, scalar: int):
    if not(scalar > 0):
        raise Exception("The scalar is {}, which is not > 0".format(scalar))

    origState = entVec.pureState
    origLabels = entVec.subSystemLabels
    origLabelArray = bitmap2Array(origLabels)
    updatedLabels = array2Bitmap(scalar * origLabelArray)

    curEntVec = copy.deepcopy(entVec)
    curState = curEntVec.pureState

    for i in range(scalar-1):
        curState = np.kron(curState, origState)

    return(curState, updatedLabels)
    

def compEntropyVec_PureState(pureState, bitPartitions, returnLabels=False):

    numSubSystems = len(bitPartitions)
    subSystemLabels = list(range(numSubSystems))
    # Here we are counting less combinations than the total possible
    # number of combinations required, because of symmetries
    # in the entropies of the reduced density matrices of
    # pure states
    numCombs = (2**(numSubSystems-1)) - 1
    systemCombs = [[] for _ in range( numCombs )]
    systemCombIndex = 0

    c_floor = math.floor((numSubSystems - 1) / 2)

    systemCombIndex = 0

    for combSize in range(c_floor):
        for subset in itertools.combinations(subSystemLabels, combSize+1):
            systemCombs[systemCombIndex] = list(subset)
            systemCombIndex = systemCombIndex + 1

    if(numSubSystems % 2 == 0):
        # Need to do a special computation for the even N's, 
        # because for the odd number, the subsystems of size
        # [up to] (N-1)/2 cover all the possible combinations
        # of subsystems on the other side of (N-1)/2, 
        # while the even ones want to have exactly half
        # of the subsystems of size N/2, since the other
        # half are redundant

        # An odd example:
        # N=3 and subsystems A,B,C
        # sA,sB,sC also cover sBC, sAC, sAB

        # An even example: 
        # N=4 and subsystems A,B,C,D.
        # sA, sB, sC, sD also cover sBCD, sACD, sABD, sABC
        # sAB, sAC, sAD also cover sCD, sBD, sBC

        # Therefore, we only need to generate the 
        # combinations of size N/2, but for a specific
        # system fixed to be in the combinations, since
        # this will cover all the ones with that system
        # not in the combinations (i.e. the other half)

        # Arbitrarily we fix the first system, but
        # any other system could be fixed instead
        nonFixedSubSys = subSystemLabels[1:]

        for unFixedSubset in itertools.combinations(nonFixedSubSys, c_floor):
            targComb = [0] + list(unFixedSubset)
            systemCombs[systemCombIndex] = targComb
            systemCombIndex = systemCombIndex + 1

    ## Computing the entropy vector

    # Generating an appropriate density operator
    if(isinstance(pureState, np.ndarray)):
        # If the pure state is an np.array, then
        # we convert it to a matrix to make things easier
        pureStateMat = np.matrix(pureState)
    else:
        pureStateMat = pureState

    densityOperator = pureStateMat @ pureStateMat.H

    entropyVec = np.full(shape=(numCombs,), fill_value=np.nan)
    for targCombInd in range(numCombs):
        targComb = systemCombs[targCombInd]

        systemsToTraceOut = list(set(subSystemLabels) - set(targComb))

        bitsToTraceOut = []
        for targSystem in systemsToTraceOut:
            bitsToTraceOut = bitsToTraceOut + bitPartitions[targSystem]
        if (targComb ==  subSystemLabels):
            subSystemEntropy = vnEntropy(densityMat=densityOperator)
        else:
            reducedSystem = partialTrace_TargBits(targDensityMat=densityOperator, bitsToTraceOut=bitsToTraceOut, indexing=0)

            subSystemEntropy = vnEntropy(densityMat=reducedSystem)

        entropyVec[targCombInd] = subSystemEntropy
    
    if(returnLabels):
        return(entropyVec, systemCombs)
    else:
        return(entropyVec)


def compEntropyVec_MixedState(densityOp, bitPartitions, returnLabels=False):
    ## Generating the possible pairings
    numSubSystems = len(bitPartitions)
    subSystemLabels = list(range(numSubSystems))
    numCombs = (2**numSubSystems) - 1

    systemCombs = [[] for _ in range( numCombs )]
    systemCombIndex = 0
    # Iteration code taken from: 
    # https://stackoverflow.com/questions/464864/how-to-get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
    for combSize in range(numSubSystems):
        for subset in itertools.combinations(subSystemLabels, combSize+1):
            systemCombs[systemCombIndex] = list(subset)
            systemCombIndex = systemCombIndex + 1
    
    ## Computing the entropy vector
    entropyVec = np.full(shape=(numCombs,), fill_value=np.nan)
    for targCombInd in range(numCombs):
        targComb = systemCombs[targCombInd]

        systemsToTraceOut = list(set(subSystemLabels) - set(targComb))

        bitsToTraceOut = []
        for targSystem in systemsToTraceOut:
            bitsToTraceOut = bitsToTraceOut + bitPartitions[targSystem]
        if (targComb ==  subSystemLabels):
            subSystemEntropy = vnEntropy(densityMat=densityOp)
        else:
            reducedSystem = partialTrace_TargBits(targDensityMat=densityOp, bitsToTraceOut=bitsToTraceOut, indexing=0)

            subSystemEntropy = vnEntropy(densityMat=reducedSystem)

        entropyVec[targCombInd] = subSystemEntropy
    if(returnLabels):
        return(entropyVec, systemCombs)
    else:
        return(entropyVec)

def bitmap2Array(subSystemBitmap: list[list[int]], subSystemLabels=None):
    """
    Converts the bitmap that is our standard method of assigning 
    qubits to a system, into an array that has the appropriate label
    at each of the indices
    
    :param subSystemBitmap: List of lists that associates bits to
                            subsystems

    Examples: 
        >>> subSystemBitmap1 = [[0,1], [2,3], [4,5]]
        >>> subSysStr1 = subSysBitmap2Str(subSystemBitmap1)
        >>> print(subSysStr1)
        Output: [0,0,1,1,2,2]

        >>> subSystemBitmap2 = [[0,1], [3,5], [2,4]]
        >>> subSysStr2 = subSysBitmap2Str(subSystemBitmap2)
        >>> print(subSysStr2)
        Output: [0,0,2,1,2,1]
    """
    numSubSystems = len(subSystemBitmap)

    if(subSystemLabels == None):
        subSystemLabels_Usable = list(range(numSubSystems))
    else:
        subSystemLabels_Usable = subSystemLabels

    numBits = np.sum([len(targList) for targList in subSystemBitmap])


    subSystem_OrderedArray = ["a" for _ in range(numBits)]
    for targSubSystemLabel in subSystemLabels_Usable:
        for targIndex in subSystemBitmap[targSubSystemLabel]:
            subSystem_OrderedArray[targIndex] = str(targSubSystemLabel)
    
    return(subSystem_OrderedArray)

def array2Bitmap(bitmapArray: list[str]):
    """
    Reverse function for bitmap2Array
    
    :param bitmapArray: Array of values that correspond to the subsystem of
                        each bit
    :type bitmapArray: list[str]
    """

    
    subSystemLabels = []
    for targBitLabel in bitmapArray:
        if not (targBitLabel in subSystemLabels):
            subSystemLabels.append(targBitLabel)
        
    numSubSystems = len(subSystemLabels)
    subSystemBitmap = [[] for _ in range(numSubSystems)]

    for targInd in range(len(bitmapArray)):
        targBitLabel = bitmapArray[targInd]
        subSysLabelIndex = subSystemLabels.index(targBitLabel)

        updatedBitmapArray = subSystemBitmap[subSysLabelIndex]
        updatedBitmapArray.append(targInd)
        subSystemBitmap[subSysLabelIndex] = updatedBitmapArray

    return(subSystemBitmap)

    

def solveCVP_coeffs(targVec, integerBasis):
    # Solve the Closest Vector Problem
    # for the given vector and lattice

    basis_Formatted = IntegerMatrix.from_matrix(integerBasis)
    basis_LLL = LLL.reduction(basis_Formatted)

    closestVector = CVP.closest_vector(basis_LLL, targVec)

    basis_np = np.matrix(integerBasis, dtype=int)
    closestVec_np = np.array(closestVector)

    coeffs = np.round(np.linalg.solve(basis_np.T, closestVec_np.T)).astype(int)


    if not(  np.array_equal(np.array(basis_np.T @ coeffs)[0,:], closestVec_np)  ):
        raise Exception("Incorrect coefficients, these do not recover closest vector")


    return(coeffs)


def latticeAprox_PureEntVec(targEntVec: np.ndarray, entropyVectorList: list[entropyVector_PureState]):
    # For now this only works with integer target vectors, and when the basis generated by the 
    # entropy vectors also only has integer coefficients, and is full rank
    
    n = targEntVec.shape[0]

    if(np.any(targEntVec < 0)): 
        raise Exception("Targ. entropy vector has a negative coefficient in it")
    if not (len(entropyVectorList) == n):
        raise Exception("To use this function you need to have the entropy vector basis being full rank")
    if(np.all(targEntVec == 0)):
        raise Exception("Entropy vector is all zeros (invalid)")

    # Converting from a list of entropy vectors to a numpy basis
    basisNp = np.full(shape=(n,n), fill_value=np.nan, dtype=int)

    for targEntVecIndex in range(len(entropyVectorList)):
        targEntVecClass = entropyVectorList[targEntVecIndex]
        basisNp[:,targEntVecIndex] = targEntVecClass.entVec
    
    coeffs = solveCVP_coeffs(targVec=targEntVec, integerBasis=basisNp)
    
    # basis_fpylll = IntegerMatrix.from_matrix(basisNp)
    # basis_lll = LLL.reduction(basis_fpylll)
    # # Getting the closest vector in the lattice to our target vector
    # closestVec = np.ndarray(CVP.closest_vector(basis_lll, targEntVec))
    # # Solving the reverse equation to get the coefficients in the entropy
    # # vector basis
    # coeffs = np.round(np.linalg.solve(basisNp.T, closestVec.T)).astype(int)

    if(np.any(coeffs) < 0):
        raise Exception("At least one coefficient is negative, which shouldn't happen but has")

    # TODO: Need to use the coefficients in this basis to construct a valid pure state 
    # TODO: (and thus entropy vector)

    for targCoeffIndex in range(len(entropyVectorList)):
        targCoeff = coeffs[targCoeffIndex]
        targEntVec = entropyVectorList[targCoeffIndex] # type: ignore

        if not (targCoeff == 0):
            targEntVec_Scaled = targCoeff * targEntVec

            if not ('curEntVec' in vars()):
                curEntVec = targEntVec_Scaled
            else:
                curEntVec = curEntVec + targEntVec_Scaled            
            

    return(curEntVec)





def gen2PartiteState(S, iterations=50):
    """
    Algorithm that will generate a pure state which can be decomposed into
    two pieces so that S(ρ_A) =  S(ρ_B) = S

    Additionally, this algorithm will produce a pure state in the lowest dimension
    that the entropy can be achieved

    :param S: von Neuman entropy of the pure state
    """

    ## Our first goal is to construct a density operator with the 
    ## specified entropy, so that we just need to purify it
    d = math.ceil(2**S)
    if(d == 1):
        # Solving for μ immeadiately, because we don't need λ's
        mu = newtonsMethod_EntropyFunctional(d=d, r=S)
        leftover = 1 - mu
        eigVals = np.array([mu, leftover])
    elif(d > 1):
        lambdaVals = 1/d
        # Computing the remainder term
        r = S - ((d-2) * h(lambdaVals))

        # Solving for the value of μ that will cause
        # the entropy of the state to most closely match S
        mu = newtonsMethod_EntropyFunctional(d=d, r=r)
        leftover = (2/d) - mu

        eigVals = np.zeros((d,))
        eigVals[0:-2] = lambdaVals
        eigVals[-2] = mu
        eigVals[-1] = leftover

    densityMat = np.diag(eigVals)

    pureStateVec = purifyMixedState(densityMat=densityMat)

    return(pureStateVec)





if __name__ == "__main__":

    ## Testing for the maximally mixed state across 3 subsystems
    densityMat = (1/16) * np.eye(16)
    bitPartitions = [[0], [1], [2,3]]
    
    maximallyMixedState_EntVec = entropyVector_MixedState(densityMat=densityMat, bitPartitions=bitPartitions)



    ## Testing that it works for a pure state as well
    maximallyEntangledState = (1/math.sqrt(2)) * np.matrix([[1], [0], [0], [1]])
    maximallyEntangledDensityOperator = maximallyEntangledState @ maximallyEntangledState.H
    
    maximallyEntangled_EntVec = entropyVector_MixedState(densityMat=maximallyEntangledDensityOperator, bitPartitions=[[0],[1]])



    ## Testing the pure state entropy vector computation on a mixed state that should generate the entropy vector
    ## (2,1,1)
    ## The state is 1/2 (|0000> + |0101> + |1010> + |1111>)
    specificPureState = 1/2 * (qudit(4,0) + qudit(4,5) + qudit(4,10) + qudit(4,15))
    subSystPartitions = [[0,1],[2],[3]]

    specificPureState_EntVec = entropyVector_PureState(pureState=specificPureState, bitPartitions=subSystPartitions)



    ## Testing entropy vector addition and multiplication
    v1 = qudit(3,0) + qudit(3,3)
    v2 = qudit(3,0) + qudit(3,5)
    v3 = qudit(3,0) + qudit(3,6)
    labels = [[0],[1],[2]]
    entVec1 = entropyVector_PureState(v1, labels)
    entVec2 = entropyVector_PureState(v2, labels)
    entVec3 = entropyVector_PureState(v3, labels)

    entVec_Sum = entVec1 + entVec2
    entVec_Mult = entVec3 * 2



    ## Testing the lattice approximation algorithm
    targEntropyVector = np.array([2,2,1], dtype=int)
    # {entVec1, entVec2, entVec3} in combination form a basis for E_3, 
    # the even integer lattice in 3 dimensions
    entVecBasisArray = [entVec1, entVec2, entVec3]
    closestEntVec = latticeAprox_PureEntVec(targEntVec=targEntropyVector, entropyVectorList=entVecBasisArray)


    ## Testing the algorithm for generating a 2 partite state
    s = 1.8
    pureState = gen2PartiteState(S=s)


    print("This is here to catch a break point")