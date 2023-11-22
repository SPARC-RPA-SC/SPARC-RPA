#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "chebyshevFiltering.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1) return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    // Sternheimer equation should be solved in RPA_OBJ, in other words, it should not use pSPARC.
    // To generate Hamiltonian in RPA domain topologies, it is necessary to generate Laplacian*Vec independent of pSPARC.
    // Then it can be called easily by any other feature. The transformation is done in function prepare_Hamiltonian(&SPARC, &RPA)
    double testResult = test_Hx(pSPARC);
}

double test_Hx(SPARC_OBJ *pSPARC) { // make a test to see the accuracy of eigenvalues and eigenvectors saved in every nuChi0Eigscomm
    return 0.0;
}