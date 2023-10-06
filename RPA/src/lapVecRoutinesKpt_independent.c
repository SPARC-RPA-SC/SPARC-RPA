// Sternheimer equation should be solved in RPA_OBJ, in other words, it should not use pSPARC.
// To generate Hamiltonian in RPA domain topologies, it is necessary to generate Laplacian*Vec independent of pSPARC.
// Then it can be called easily by any other feature.