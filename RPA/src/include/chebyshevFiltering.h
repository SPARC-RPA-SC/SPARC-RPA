#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"
#include "main.h"

void test_Hx_nuChi0(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults);

void sternheimer_solver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, int printFlag);

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex);

double find_min_eigenvalue(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm);

#endif