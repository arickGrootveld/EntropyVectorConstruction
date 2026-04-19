import cvxpy as cvp
import numpy as np

def miqpApprox(aprxVec, basisVecs, verbose=False):
    '''
    Inputs:
        aprxVec: The vector to approximate
        basisVecs: A matrix who's columns are the vectors we will use to approximate aprxVec
        verbose=False: Parameter of the solver
    Output:
        z_opt: The value of z that minimizes ||Az - y||_2 subject to the constraint 
               that z must be a vector of non-negative integers
    '''
    (vecSize, numVecs) = basisVecs.shape
    z = cvp.Variable(numVecs, integer=True)
    objective = cvp.Minimize(cvp.sum_squares((basisVecs @ z) - aprxVec))
    constraints = [z >= 0]
    problem = cvp.Problem(objective, constraints)
    result = problem.solve(solver=cvp.SCIP, verbose=verbose) 
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise Exception(f"Solver did not return an optimal solution. Status: {problem.status}")
    z_opt = np.rint(z.value).astype(int)
    return(z_opt)

def entVecAprx(entVecToAprx, useableEntVecs):
    '''
    Inputs:
        aprxEntVec: A numpy array containing an entropy vector that we want to approximate
        useableEntVecs: A list containing entropy vectors that we can combine to approximate
                        aprxEntVec
    '''

    numEntVecs = len(useableEntVecs)
    aprxEntVecLen = entVecToAprx.shape[0]

    # Filling in the basis vector matrix, so we can solve the mixed integer quadratic program
    entVecBasisMat = np.full(shape=(aprxEntVecLen,numEntVecs), fill_value=np.nan)
    for index in range(len(useableEntVecs)):
        targEntVec = useableEntVecs[index]
        targEntVec_Array = targEntVec.entVec
        entVecBasisMat[:,index] = targEntVec_Array

    coeffsOfApprox = miqpApprox(aprxVec=entVecToAprx, basisVecs=entVecBasisMat)

    aprxEntVec = None
    for coeffInd in range(coeffsOfApprox.shape[0]):
        
        targCoeff = coeffsOfApprox[coeffInd]
        if(targCoeff > 0):
            # If the coefficient is non-zero, then we will use the entropy vector
            targEntVec = useableEntVecs[coeffInd]
            scaledEntVec = targCoeff * targEntVec
            if(aprxEntVec == None):
                aprxEntVec = scaledEntVec
            else:
                aprxEntVec = aprxEntVec + scaledEntVec

    return(aprxEntVec)
            





if(__name__ == "__main__"):
    from entropyVectorAlgorithms import entropyVector_MixedState

    ## Testing the miqpApprox function
    B = np.matrix([[1.1,1,0,1], [0,1,1,1], [1,0,1,1]])
    y = np.matrix([[1],[2],[3]])
    optimalSolution = miqpApprox(aprxVec=y, basisVecs=B)

    ## Testing the entVecAprx function

    # u1 corresponds to the maximally entangled state
    u1DensMat = (1/2) * np.matrix([[1, 0, 0, 1], [0,0,0,0], [0,0,0,0], [1,0,0,1]])
    u1EntVec = entropyVector_MixedState(densityMat=u1DensMat, bitPartitions=[[0], [1]])
    # u2 corresponds to the maximally mixed state on B, and A being degenerate
    u2DenseMat = np.kron(np.matrix([[1,0], [0,0]]), (1/2) * np.eye(2) )
    u2EntVec = entropyVector_MixedState(densityMat=u2DenseMat, bitPartitions=[[0], [1]])
    # u3 corresponds to the maximally mixed state on A, and B being degenerate
    u3DenseMat = np.kron((1/2) * np.eye(2), np.matrix([[1,0], [0,0]]) )
    u3EntVec = entropyVector_MixedState(densityMat=u3DenseMat, bitPartitions=[[0], [1]])
    # u4 corresponds to a A and B being entirely classically correlated
    u4DenseMat = (1/2) * np.matrix([[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1]])
    u4EntVec = entropyVector_MixedState(densityMat=u4DenseMat, bitPartitions=[[0], [1]])

    entVecsToUse = [u1EntVec, u2EntVec, u3EntVec, u4EntVec]
    entVecToAprox = y

    aprxEntVec = entVecAprx(entVecToAprx=entVecToAprox, useableEntVecs=entVecsToUse)

    


    print("This is here to catch a break(point)")


