#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "finalization.h"
#include "kroneckerLaplacian.h"

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
    free(pRPA->RRoriginSeqEigs);
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
        if (pSPARC->isGammaPoint) {
            free(pRPA->deltaVs);
        } else {
            free(pRPA->deltaVs_kpt);
        }
        int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
        if (!flagNoDmcomm) {
            if (pSPARC->isGammaPoint) {
                free(pRPA->nearbyBandIndicesGamma);
                free(pRPA->neighborBandIndicesGamma); // free NULL pointer is fine
                free(pRPA->neighborBandsGamma);
                free(pRPA->allEpsilonsGamma);
                free(pRPA->deltaRhos);
                free(pRPA->sprtNuDeltaVs);
                free(pRPA->Ys);
                free(pRPA->deltaPsisReal);
                free(pRPA->deltaPsisImag);

                free(pSPARC->pois_const);
                free_kron_Lap(pSPARC->kron_lap_exx);
                free(pSPARC->kron_lap_exx);
                if (pRPA->flagCOCGinitial) {
                    free(pRPA->allXorb);
                    free(pRPA->allLambdas);
                }
            } else {
                free(pRPA->deltaRhos_kpt);
                free(pRPA->sprtNuDeltaVs_kpt);
                free(pRPA->Ys_kpt);
                free(pRPA->deltaPsis_kpt);
                if (pRPA->flagCOCGinitial) {
                    free(pRPA->allXorb_kpt);
                    free(pRPA->allLambdas);
                }
            }
        }
        Free_scfvar(pSPARC);
    }
    Finalize(pSPARC);
}
