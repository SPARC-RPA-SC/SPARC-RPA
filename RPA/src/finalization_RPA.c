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
    // free eigs
    free(pRPA->RRnuChi0Eigs);
    free(pRPA->RRnuChi0EigVecs);
    free(pRPA->ErpaTerms);
    // free communicators
    MPI_Comm_free(&pRPA->nuChi0Eigscomm);
    MPI_Comm_free(&pRPA->nuChi0EigsBridgeComm);
    MPI_Comm_free(&pRPA->nuChi0BlacsComm);
    if (pSPARC->isGammaPoint) {
        free(pRPA->Hp);
        free(pRPA->Mp);
        free(pRPA->Q);
    } else {
        free(pRPA->Hp_kpt);
        free(pRPA->Mp_kpt);
        free(pRPA->Q_kpt);
    }
    if (pRPA->nuChi0EigscommIndex != -1) {
        if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
            if (pSPARC->isGammaPoint) {
                free(pRPA->deltaRhos_phi);
                free(pRPA->deltaVs_phi);
                free(pRPA->Ys_phi);
            } else {
                free(pRPA->deltaRhos_kpt_phi);
                free(pRPA->deltaVs_kpt_phi);
                free(pRPA->Ys_kpt_phi);
            }
        }
        int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
        if (!flagNoDmcomm) {
            if (pSPARC->isGammaPoint) {
                free(pRPA->nearbyBandIndicesGamma);
                free(pRPA->neighborBandIndicesGamma); // free NULL pointer is fine
                free(pRPA->neighborBandsGamma);
                free(pRPA->allEpsilonsGamma);
                free(pRPA->deltaRhos);
                free(pRPA->deltaVs);
                free(pRPA->deltaPsisReal);
                free(pRPA->deltaPsisImag);
            } else {
                free(pRPA->deltaRhos_kpt);
                free(pRPA->deltaVs_kpt);
                free(pRPA->deltaPsis_kpt);
            }
        }
        Free_scfvar(pSPARC);
    }
    Finalize(pSPARC);
}
