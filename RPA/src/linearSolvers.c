#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"

#include "linearSolver.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double, double, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts, int maxIter, double *resNormRecords) {

    return 0;
}

int kpt_solver(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex deltaPsis_kpt, 
     double _Complex *SternheimerRhs_kpt, int nuChi0EigsAmounts, int maxIter, double *resNormRecords) {

     return 0;
}