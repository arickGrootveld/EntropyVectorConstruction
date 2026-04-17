import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct as spTensProd
import scipy as scp

import math

## Utility functions to use for the project

def genRandPureState(dim, rngSeed=-1):
    if(rngSeed<= 0):
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=rngSeed)

    cmplxGaussSamp = rng.normal(loc=0, scale=1, size=(dim,1)) + 1j*rng.normal(loc=0, scale=1, size=(dim, 1))

    normalizedCmplxGauss = cmplxGaussSamp / np.linalg.norm(cmplxGaussSamp, ord=2)

    return(normalizedCmplxGauss)


def genPureStateDMats(dim, num=1, rngSeed=-1):
    pureStateDMats = np.full(shape=(dim, dim, num), fill_value=np.nan, dtype=np.complex128)

    for pStInd in range(num):
        if(rngSeed <= 0):
            targPSt = genRandPureState(dim=dim, rngSeed=rngSeed)
        else:
            targPSt = genRandPureState(dim=dim, rngSeed=rngSeed+pStInd)

        targPStDMat = np.matrix(targPSt) @ np.matrix(targPSt).H
        pureStateDMats[:,:,pStInd] = targPStDMat

    return(pureStateDMats)

def genRandDensityMat(dim, rngSeed=-1):

    # Setting up the random number generation
    if(rngSeed <= 0):
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=rngSeed)


    # Generating N samples from a N dimensional complex Gaussian to use for the Haar basis
    complexGaussianSamples = rng.normal(loc=0, scale=1, size=(dim, dim)) + 1j*rng.normal(loc=0, scale=1, size=(dim, dim))
    
    normalizedCmplxSamps = complexGaussianSamples / np.linalg.norm(complexGaussianSamples, axis=0)

    basisVectors = gram_schmidt(A=normalizedCmplxSamps)

    dirichletSample = rng.dirichlet(alpha=np.ones(shape=(dim,)), size=(1,))

    randDensityMat = np.zeros(shape=(dim, dim), dtype=basisVectors.dtype)

    for basisIndex in range(basisVectors.shape[0]):
        targBasisVec = np.matrix(basisVectors[:,basisIndex]).H
        targProb = dirichletSample[0,basisIndex]
        targRankOneMat = targBasisVec @ targBasisVec.H
        randDensityMat = randDensityMat + targProb * targRankOneMat

    return randDensityMat

# Inspired by: https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
def gram_schmidt(A):
    '''
    input: A set of linearly independent vectors stored
              as the columns of matrix A
       outpt: An orthongonal basis for the column space of A
       '''
    # get the number of vectors.
    
    n = A.shape[1]
    basis = np.full(shape=(n,n), fill_value=np.nan, dtype=A.dtype)

    for ind in range(n):
        if(ind == 0):
            # First vector we normalize and then accept
            basis[:,ind] = A[:,0] / np.linalg.norm(A[:,0])

        else:
            curVec = A[:,ind]
            for backwardsInd in range(ind):

                intermediateVar = (basis[:,backwardsInd] * np.dot(A[:,ind], np.conj(basis[:,backwardsInd])))

                curVec = curVec - intermediateVar
            curVec = curVec / np.linalg.norm(curVec)
            basis[:,ind] = curVec

    return(basis)


def genDensMats(dim, num=1, rngSeed=-1):
    densityMats = np.full(shape=(dim,dim, num), fill_value=np.nan, dtype=np.complex128)

    for densMatInd in range(num):
        if(rngSeed <= 0):
            targDenseMat = genRandDensityMat(dim=dim, rngSeed=rngSeed)
        else:
            # Incrementing the rng seed so the resulting density matrices aren't always the same
            targDenseMat = genRandDensityMat(dim=dim, rngSeed=rngSeed+densMatInd)

        densityMats[:,:, densMatInd] = targDenseMat

    return(densityMats)

def qudit(d, bit):
    """
    Generate the corresponding qudit in the d dimensional Hilbert space
    
    :param d: Qubit dimension of Hilbert space (i.e. dim(H) = 2**d)
    :param bit: Bit (e.g. bit=0 yields |00...00>)
    """
    hDim = 2**d
    if(bit > hDim):
        raise Exception("Bit number is too high. d: {}, bit: {}".format(d, bit))
    targQudit = np.zeros((hDim,1))
    targQudit[bit] = 1
    return(targQudit)

def h(x, base=2):
    """
    Entropy functional: h(x) = -x log_2(x)
    
    :param x:
    :param base: base of the logarithm
    """
    if(x in [0,1]):
        # By convention we take 0 log_2(0) = 0
        return 0
    elif(np.isclose(x, 0)):
        return 0
    else:
        return( -1*x*math.log(x, base) )


def vnEntropy(densityMat):

    eigVals, _ = np.linalg.eigh(densityMat)

    entropyTotal = 0
    for eigVal in eigVals:
        entropyTotal = entropyTotal + h(np.abs(eigVal))

    return(entropyTotal)


def intToBinStr(integer, length=-1):
    # Code for conversion to binary string and filling in leading bits is taken from:
    # https://stackoverflow.com/questions/73285087/how-can-i-convert-an-integer-to-its-binary-representation-as-a-numpy-array

    if(length<= 0):
        # If length not specified, then we just keep it min length
        binStr = bin(integer)[2:]
    else:
        binStr = bin(integer)[2:].zfill(length)
    
    return(binStr)
def binStrToInt(binStr):
    return( int(binStr, 2) )

def tensorProd(mat1, mat2, mode="np"):
    '''
    This is a function that should take as input two numpy matrices (or sympy matrices),
    and will return the tensor product (or kronecker product depending on your notation)
    '''

    # Deciding what kind of objects we're dealing with
    if(mode in ["numpy, np"]):
        modeName = "numpy"
    if(mode in ["sp", "sympy"]):
        modeName = "sympy"
    else:
        modeName = "numpy"

    
    if(modeName == "numpy"):
        # print("This is here in numpy mode")
        output = np.kron(mat1, mat2)
        
    else:
        output = spTensProd(mat1, mat2)
        
    return(output)

def partialTrace_TargBits(targDensityMat, bitsToTraceOut, mode="np", indexing=1):
    ''' 
    This is a function that takes as input a density matrix, and a set of bits of the 
    matrix that we would like to trace out, and then performs the partial trace of this density operator.
    IMPORTANT: This function assumes the bitsToTraceOut vector is using 1 indexed by default

    targDensityMat: should be a numpy or sympy (depending on mode variable) matrix
                    that we want to perform the partial trace of

    bitsToTraceOut: should be a list or tuple of integers that indicates
                    what indices we want to trace out 
                    (assumed to be 1 indexed)

    mode:           specifies whether the density matrix is a numpy or sympy matrix
    '''
    # Deciding what kind of objects we're dealing with
    if(mode in ["numpy, np"]):
        modeName = "numpy"
    if(mode in ["sp", "sympy"]):
        modeName = "sympy"
    else:
        modeName = "numpy"

    qubitDim = int(np.log2(targDensityMat.shape[0]))

    # Creating a list of dimensions that is 1 indexed for readability
    if(indexing == 1):
        listOfInds = np.arange(qubitDim) + 1
    elif(indexing == 0):
        listOfInds = np.arange(qubitDim)
    else:
        raise Exception("We only allow 0 and 1 indexing, but the indexing input was {}".format(indexing))

    d1 = len(bitsToTraceOut)

    d2 = qubitDim - d1

    # Pre-allocating the reduced density matrix
    if(modeName == "sympy"):
        reducedDensityMat = sp.Matrix(np.zeros((2**d2, 2**d2)))
    else:
        reducedDensityMat = np.zeros((2**d2, 2**d2))

    for binaryNumInd in range((2**d1)):

        targBinaryString = intToBinStr(binaryNumInd, length=d1)

        # Setting up a counter to track where in the binary string we are, 
        # as we iterate over all possible qubit indices
        removeCounter = 0
        
        # Pre-defining the left and right matrices we will be using in our
        # matrix multiplications
        # (sympy supports mat mul between np mats and sp mats, so just using np mat here)
        targLeftMat = np.matrix([[1]])
        targRightMat = np.matrix([[1]])
   

        for qubitSysInd in range(qubitDim):
            
            # If its a bit in the system that we want to remove, 
            # then we iterate through the binary vector, and 
            # select the appropriate vector based on the binary value
            if(listOfInds[qubitSysInd] in bitsToTraceOut):
                targBit = targBinaryString[removeCounter]
                removeCounter = removeCounter + 1
                # If the bit is 0, want to use <0| for left and |0> for right
                if(targBit == "0"):
                    curLeftMat = np.matrix([[1,0]], dtype=int)
                    curRightMat = np.matrix([[1],[0]], dtype=int)
                # If bit is 1, want to use <1| for left and |1> for right
                elif(targBit == "1"):
                    curLeftMat = np.matrix([[0,1]], dtype=int)
                    curRightMat = np.matrix([[0], [1]], dtype=int)
                else:
                    raise Exception("Something went very wrong that should not have gone wrong. targBit is: {}".format(targBit))
            
            # If its not a bit to remove, then use an identity matrix in the tensor product
            else:
                
                curLeftMat = np.eye(2, dtype=int)
                curRightMat = np.eye(2, dtype=int)
            

            targLeftMat = tensorProd(targLeftMat, curLeftMat, mode="np")
            targRightMat = tensorProd(targRightMat, curRightMat, mode="np")


        # Performing the left and right multiplications
        curMatMultResult = targLeftMat @ targDensityMat @ targRightMat

        reducedDensityMat = reducedDensityMat + curMatMultResult


    return(reducedDensityMat)


def purifyMixedState(densityMat):


    d = densityMat.shape[0]

    eigVals, eigVecs = np.linalg.eigh(np.matrix(densityMat))

    targPureState = np.zeros(shape=(d**2,1), dtype=complex)

    for eInd in range(eigVals.shape[0]):
        targEigVal = eigVals[eInd]
        targEigVec = eigVecs[:,eInd]

        doubledEVec = math.sqrt(targEigVal) * np.kron(targEigVec, targEigVec)

        targPureState = targPureState + doubledEVec

    return(targPureState)


def newtonsMethod_EntropyFunctional(d, r):
    """
    Nevermind, I got Newtons method working
    
    :param d: dimension
    :param r: remainder
    """
    entFunc = lambda x : h(x) + h((2/d) - x) - r
    entFuncDeriv = lambda x : math.log2((2/(d*x)) - 1)

    midpoint = 3 / (2*d)

    newtonResult = scp.optimize.newton(func=entFunc, x0=midpoint, fprime=entFuncDeriv)

    return(newtonResult)

def successiveApprox_EntropyFunctional(d, r, numSteps=100):
    """
    A solver for the entropy functional that we want. 
    (Originally tried Newtons method, but that had domain issues)
    This only works because the derivative is strictly negative
    so the solution in the interval should be unique

    :param d: dimension of problem
    :param r: remainder term (we are trying to match this)
    """

    entFunc = lambda x : h(x) + h((2/d) - x)

    curLb = 1/d
    curUb = 2/d

    i = 0

    while i < numSteps:
        midPoint_x = (curLb + curUb) / 2
        curOutput = entFunc(midPoint_x)

        if(curOutput > r):
            curLb = midPoint_x
        elif(curOutput < r):
            curUb = midPoint_x
        else:
            # This only happens if we get equality,
            # at which point we want to return that midpoint
            i = numSteps
        i = i + 1
    
    return(midPoint_x)



if(__name__ == "__main__"):


    # Checking to make sure the purification and partial trace code work
    dMat = genRandDensityMat(dim=4, rngSeed=1020588)
    pureState = purifyMixedState(dMat)
    pureStateDMat = pureState @ pureState.H # type: ignore
    reducDMat = partialTrace_TargBits(pureStateDMat, bitsToTraceOut=[3,4])

    assert np.allclose(reducDMat, dMat), "Either the partial trace or purification code is not working"


    # Testing solver for the entropy functional
    d = 3
    r = 0.6092

    output = successiveApprox_EntropyFunctional(d=d, r=r, numSteps=50)

    print("This is here to catch a break point")