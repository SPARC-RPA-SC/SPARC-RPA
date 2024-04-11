#ifndef EIGKPTRPA
#define EIGKPTRPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double minEig, double maxEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

void YT_multiply_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, int flagNoDmcomm, int printFlag);

void Y_orth_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int DMnd, int Nspinor_eig, int printFlag);

void project_YT_nuChi0_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int flagNoDmcomm, int printFlag);

void generalized_eigenproblem_solver_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, MPI_Comm blacsComm, int blksz, int *signImag, int printFlag);

void subspace_rotation_unify_eigVecs_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int printFlag);

double evaluate_cheFSI_error_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm);
#endif