#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults);

#endif