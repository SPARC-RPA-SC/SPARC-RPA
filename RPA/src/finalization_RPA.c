#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "finalization.h"

void finalize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    
    Finalize(pSPARC);
    // free q-points
    free(pRPA->qptWts);
    free(pRPA->q1);
    free(pRPA->q2);
    free(pRPA->q3);
    // free omega
    free(pRPA->omega);
    free(pRPA->omega01);
    free(pRPA->omegaWts);
}