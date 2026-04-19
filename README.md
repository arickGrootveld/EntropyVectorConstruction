# Entropy Vector Construction
Algorithms for constructing mixed and pure states from entropy vectors. Work done in conjunction with Jason Pollack while at Syracuse University. 

The current algorithm only allows full construction for pure states with two partitions, and for pure states with three partitions but with the entropy vector being non-negative integer valued. 

Currently the three partition version will break for non-valid entropy vectors (one's not satisfying Araki-Lieb inequality) and for entropy vectors with non-whole numbers as entries. This is the next step for the algorithm. 

## File structure
- 'Demo_PureStateFromEntropyVector.ipynb' contains a demonstration of using the code to construct a pure state for a given entropy vector
with three partitions. 

- 'utilities.py' contains a bunch of extra functions. Some stuff should be added, and some stuff should be removed from this (namely, fpylll is only needed for the approximation method for arbitary partitions)

- 'threePartitionAlg.py' implements the three partition version of the algorithm

- 'entropyVectorAlgorithms.py' contains more algorithms for constructing a state to match an entropy vector, along with classes that standardize entropy vectors. The two most interesting algorithms in here: one generates a mixed state with a specified entropy in the smallest possible number of dimensions, and the other approximates a target entropy vector using integer combinations of a set of basis entropy vectors using a lattice shortest vector algorithm.

- 'mixedIntegerApprox.py' has an implementation of a mixed integer quadratic program to approximate a given entropy vector from a collection of given states

