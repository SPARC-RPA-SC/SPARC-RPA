#ifndef EIGGAMMARPA
#define EIGGAMMARPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, int omegaIndex, int flagNoDmcomm, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int isGammaPoint, int printFlag);
#endif