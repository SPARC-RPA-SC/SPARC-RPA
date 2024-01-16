#ifndef EIGKPTRPA
#define EIGKPTRPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double minEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

void YT_multiply_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

void project_YT_nuChi0_Y_kpt(RPA_OBJ* pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int isGammaPoint, int printFlag);
#endif