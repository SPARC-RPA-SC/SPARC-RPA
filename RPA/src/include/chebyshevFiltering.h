#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults);

void test_chi0_times_deltaV(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double *sternSolverAccuracy);

double test_chi0_times_deltaV_gamma(SPARC_OBJ *pSPARC, int sg, double epsilon, double omega, double *deltaPsis, double *deltaVs, double *psi, int nuChi0EigsAmounts);

double test_chi0_times_deltaV_kpt(SPARC_OBJ *pSPARC, int sg, int kPq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi, int nuChi0EigsAmounts);

int find_kPq(int Nkpts, int **kPqList, int kPqSym);

#endif