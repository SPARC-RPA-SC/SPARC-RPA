#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "sternheimerEquation.h"
#include "electrostatics_RPA.h"

void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs_phi, double *nuChi0DVs_phi, int nuChi0EigsAmount, int flagNoDmcomm) {
    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
        Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, DVs_phi + nuChi0EigIndex*pSPARC->Nd_d, pRPA->deltaVs + nuChi0EigIndex*pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
    }
    if (!flagNoDmcomm) {
        sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, 0);
    }
    collect_transfer_deltaRho_gamma(pSPARC, pRPA->deltaRhos, pRPA->deltaRhos_phi, nuChi0EigsAmount, 0, pRPA->nuChi0Eigscomm);
    Calculate_deltaRhoPotential_gamma(pSPARC, pRPA->deltaRhos_phi, nuChi0DVs_phi, nuChi0EigsAmount, 0, pRPA->deltaVs, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
}

void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt_phi, double _Complex *nuChi0DVs_kpt_phi, int nuChi0EigsAmount, int flagNoDmcomm) {
    for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
        Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, DVs_kpt_phi + nuChi0EigIndex*pSPARC->Nd_d, pRPA->deltaVs_kpt + nuChi0EigIndex*pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
    }
    if (!flagNoDmcomm) {
        sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, 0);
    }
    collect_transfer_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, pRPA->deltaRhos_kpt_phi, nuChi0EigsAmount, 0, pRPA->nuChi0Eigscomm);
    double qptx = pRPA->q1[qptIndex]; double qpty = pRPA->q2[qptIndex]; double qptz = pRPA->q3[qptIndex];
    Calculate_deltaRhoPotential_kpt(pSPARC, pRPA->deltaRhos_kpt_phi, nuChi0DVs_kpt_phi, qptx, qpty, qptz, nuChi0EigsAmount, 0, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
}