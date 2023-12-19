#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "finalization.h"

void finalize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    
    // free sym k-points
    free(pRPA->kptWts);
    free(pRPA->k1);
    free(pRPA->k2);
    free(pRPA->k3);
    // free q-points
    free(pRPA->qptWts);
    free(pRPA->q1);
    free(pRPA->q2);
    free(pRPA->q3);
    for (int nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        free(pRPA->kPqSymList[nk]);
        free(pRPA->kPqList[nk]);
        free(pRPA->kMqList[nk]);
    }
    free(pRPA->kPqSymList);
    free(pRPA->kPqList);
    free(pRPA->kMqList);
    // free omega
    free(pRPA->omega);
    free(pRPA->omega01);
    free(pRPA->omegaWts);
    // free communicators
    MPI_Comm_free(&pRPA->nuChi0Eigscomm);
    MPI_Comm_free(&pRPA->nuChi0EigsBridgeComm);
    if (pRPA->nuChi0EigscommIndex != -1) {
        if (pSPARC->isGammaPoint) {
            free(pRPA->deltaRhos);
            free(pRPA->deltaRhos_phi);
            free(pRPA->deltaVs);
            free(pRPA->deltaVs_phi);
            free(pRPA->deltaPsisReal);
            free(pRPA->deltaPsisImag);
        } else {
            free(pRPA->deltaRhos_kpt);
            free(pRPA->deltaRhos_kpt_phi);
            free(pRPA->deltaVs_kpt);
            free(pRPA->deltaVs_kpt_phi);
            free(pRPA->deltaPsis_kpt);
        }
        Free_scfvar(pSPARC);
    }
    Finalize(pSPARC);
}
