#ifndef EIGGAMMARPA
#define EIGGAMMARPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double maxEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

void Y_orth_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int DMnd, int Nspinor_eig, int printFlag);

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, MPI_Comm blacsComm, int blksz, int *signImag, int printFlag);

void automem_pdgeevx_( 
    const char *balanc, const char *jobvl, const char *jobvr, const char *sense, 
    const int *n, double *a, const int *desca, double *wr, double *wi, 
    double *vl, const int *descvl, double *vr, const int *descvr, 
    int *ilo, int *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, int *info);

void subspace_rotation_unify_eigVecs_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int printFlag);
#endif