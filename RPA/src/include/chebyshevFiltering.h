#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"
#include "main.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults);

void test_sternheimer_solver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double *sternSolverAccuracy);

#endif