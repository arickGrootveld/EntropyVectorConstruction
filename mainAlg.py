import numpy as np

from numpy import kron as tensProd


def flatten_AndDetype(aList):
    '''
    Flattens the list and sets the types to all match
    '''
    return(np.array([item for sublist in aList for item in sublist]).tolist())

def convertBetweenEntVecOrderings(entVec, curOrdering, targOrdering):
    '''
    A function that will apply a permutation to an entropy vector, so that
    the ordering of the elements matches targOrdering, when the original 
    ordering was curOrdering
    '''

    permutation = [curOrdering.index(i) for i in targOrdering]

    return entVec[permutation]


def repeatedTensorProd(dMat, k):
    '''
    Applys a repeated tensor product of the input density matrix, k times
    '''
    if(k <= 0):
        return(np.matrix([[1]]))
    if(k == 1):
        return(dMat)
    else:
        targDenseMat = np.matrix([[1]])
        for i in range(k):
            targDenseMat = tensProd(targDenseMat, dMat)
        
        return(targDenseMat)

def checkIsValidEntVec_2N(v):
    '''
    Input:
        v: An np.array of non-negative integer values

    Output: 
        isValid: Either 0 or 1 (0 means it is not a valid entropy vector)
    '''

    if not (len(v) == 3):
        raise Exception("The length of v is: {}, when it should be 3".format(len(v)))
    
    a1 = v[1] + v[2] - v[0]
    a2 = v[0] + v[1] - v[2]
    a3 = v[0] + v[2] - v[1]

    if((a1 < 0) or (a2 < 0) or (a3 < 0)):
        isValid = 0
    elif((v[0] < 0) or (v[1] < 0) or (v[2] < 0)):
        isValid = 0
    else:
        isValid = 1

    return(isValid)


def reconstIntegerVec_2N(v):
    '''
    Input: 
        v: A vector of non-negative integer values

    Output:
        hatV: The reconstructed vector
        vCoefs: An array of length 4, which specifies the coefficients 
                of u1, u2, u3, u4 (in that order) that are required to
                reconstruct v. If it is not a valid entopy vector
                this will be all -1's
    '''


    isValidEntVec = checkIsValidEntVec_2N(v)

    if(not isValidEntVec):
        # If not a valid entropy vector then set it to be all -1's
        vCoefs = [-1,-1,-1,-1]
    else:
        vCoefs = [0,0,0,0]

        # Defining the vectors that will be our basis set
        u1 = np.array([0,1,1])
        u2 = np.array([1,0,1])
        u3 = np.array([1,1,0])
        u4 = np.array([1,1,1])

        uVec = [u1, u2, u3, u4]
        vecLabels = np.array([1,2,3])

        perm = np.argsort(v)[::-1]

        vprime = v[perm]
        newVecLabels = vecLabels[perm]

        u2prime = uVec[newVecLabels[1] - 1]
        u3prime = uVec[newVecLabels[2] - 1]

        b1 = vprime[1] + vprime[2] - vprime[0]
        b2 = vprime[0] - vprime[1]
        b3 = vprime[0] - vprime[2]

        hatV = (b1 * u4) + (b2 * u2prime) + (b3 * u3prime)

        vCoefs[3] = b1
        vCoefs[newVecLabels[0]-1] = 0
        vCoefs[newVecLabels[1]-1] = b2
        vCoefs[newVecLabels[2]-1] = b3
    
    return(hatV, vCoefs)
    

def vCoeffs2DensityOperator_2N(vCoeffs):
    '''
    Input: 
        vCoeffs: An array of entries specifying the coefficients of u1, u2, u3, u4 for the reconstruction

    Output:
        densMat: A density matrix that will have the same entropy vector as v
        bitPartitions: A list containing two lists that correspond to the indices of systems in A and B
    '''

    # u1 corresponds to the maximally entangled state
    u1DensMat = (1/2) * np.matrix([[1, 0, 0, 1], [0,0,0,0], [0,0,0,0], [1,0,0,1]])

    # u2 corresponds to the maximally mixed state on B, and A being degenerate
    u2DenseMat = (1/2) * np.eye(2)

    # u3 corresponds to the maximally mixed state on A, and B being degenerate
    u3DenseMat = (1/2) * np.eye(2)

    # u4 corresponds to a A and B being entirely classically correlated
    u4DenseMat = (1/2) * np.matrix([[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1]])

    # Initializing the density matrix that we will use
    densMat = np.matrix([[1]])

    # Arrays that will track the bits in system A and system B
    aBits = []
    bBits = []

    curBitCount = 0

    if(vCoeffs[0] > 0):
        u1DensRepeated = repeatedTensorProd(dMat=u1DensMat, k=vCoeffs[0])
        densMat = tensProd(densMat, u1DensRepeated)

        # Updating the labels of the A systems and B systems based on the tensor products
        aBits.append([curBitCount + 2*i for i in range(vCoeffs[0])])
        bBits.append([curBitCount + 2*i + 1 for i in range(vCoeffs[0])])
        curBitCount = curBitCount + (2*vCoeffs[0])

    if(vCoeffs[1] > 0):
        u2DensRepeated = repeatedTensorProd(dMat=u2DenseMat, k=vCoeffs[1])
        densMat = tensProd(densMat, u2DensRepeated)
        # Since u2 corresponds to a state on B, but A being degenerate, we only need to update B
        bBits.append([curBitCount + i for i in range(vCoeffs[1])])
        curBitCount = curBitCount + vCoeffs[1]

    if(vCoeffs[2] > 0):
        u3DensRepeated = repeatedTensorProd(dMat=u3DenseMat, k=vCoeffs[2])
        densMat = tensProd(densMat, u3DensRepeated)
        # Since u3 corresponds to a state on A, but B being degenerate, we only need to update A
        aBits.append([curBitCount + i for i in range(vCoeffs[2])])
        curBitCount = curBitCount + vCoeffs[2]
    
    if(vCoeffs[3] > 0):
        u4DensRepeated = repeatedTensorProd(dMat=u4DenseMat, k=vCoeffs[3])
        densMat = tensProd(densMat, u4DensRepeated)

        # Updating the labels of the A systems and B systems based on the tensor products
        aBits.append([curBitCount + 2*i for i in range(vCoeffs[3])])
        bBits.append([curBitCount + 2*i + 1 for i in range(vCoeffs[3])])
        curBitCount = curBitCount + (2*vCoeffs[3])
    
    aBitsFlat = flatten_AndDetype(aBits)
    bBitsFlat = flatten_AndDetype(bBits)

    bitPartitions = [aBitsFlat, bBitsFlat]

    return(densMat, bitPartitions)



def entVec2DenseMat_Int2N(entVec, entVecOrdering):
    '''
    Function that maps a non-negative valued entropy vector and its ordering,
    into a density matrix with the exact same entropy vector
    '''

    if(not (len(entVec) == 3)):
        raise Exception("Entropy vector is of length: {}, but this function only works if it has length 3".format(len(entVec)))
    correctOrdering = [[0,1], 0, 1]

    reorderedEntVec = convertBetweenEntVecOrderings(entVec=entVec, curOrdering=entVecOrdering, targOrdering=correctOrdering)

    vHat, vCoeffs = reconstIntegerVec_2N(v=reorderedEntVec)

    if(not np.array_equal(vHat, reorderedEntVec)):
        raise Exception("The reordered entropy vector and the reconstructed entropy vector differ")
    

    denseMat, bitPartitions = vCoeffs2DensityOperator_2N(vCoeffs=vCoeffs)

    return(denseMat, bitPartitions)







if(__name__ == "__main__"):
    from entropyVectorAlgorithms import entropyVector_MixedState, entropyVector_PureState
    from utilities import purifyMixedState
    
    testEntVec = np.array([2,2,3])
    curBitOrdering = [0, 1, [0,1]]

    testDensMat, bitPartitions_MS = entVec2DenseMat_Int2N(entVec=testEntVec, entVecOrdering=curBitOrdering)

    # Computing the entropy vector for the generated density matrix to make sure it matches the 
    # original entropy vector
    targEntVec_MS = entropyVector_MixedState(densityMat=testDensMat, bitPartitions=bitPartitions_MS)
    print("The entropy vector of the bipartite density matrix is: {}".format(targEntVec_MS.entVec))

    # Constructing the pure state, and verifying that it has the same entropy vector as the input
    testPureState = purifyMixedState(testDensMat)
    numBits_MS = len(flatten_AndDetype(bitPartitions_MS))
    pureStateBitInds = [i + numBits_MS for i in range(numBits_MS)]
    bitPartitions_PS = bitPartitions_MS + [pureStateBitInds]
    targEntVec_PS = entropyVector_PureState(pureState=testPureState, bitPartitions=bitPartitions_PS)
    print("The entropy vector of the tripartite pure state is: {}".format(targEntVec_PS.entVec))

    

    print("This is here")
