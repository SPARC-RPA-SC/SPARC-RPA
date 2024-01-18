#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"
#include "main.h"

void test_Hx_nuChi0(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults);

void cheFSI_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex);

double find_min_eigenvalue(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm);

double compute_ErpaTerm(double *RRnuChi0Eigs, int nuChi0Neig, double omegaMesh01, double qptOmegaWeight, int *signImag);
#endif