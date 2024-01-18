#ifndef EIGGAMMARPA
#define EIGGAMMARPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

void Y_orth_gamma(RPA_OBJ* pRPA, int DMnd, int Nspinor_eig);

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int *signImag, int printFlag);

void subspace_rotation_unify_eigVecs_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int printFlag);
#endif