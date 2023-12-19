#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stddef.h"
#include "electrostatics.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "tools.h"
#include "linearSolver.h"

#include "main.h"
#include "electrostatics_RPA.h"

void collect_transfer_deltaRho(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigsAmounts = pRPA->nNuChi0Eigscomm; // only the first \Delta V in the nuChi0Eigscomm takes part in the test
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int ncol = pSPARC->Nband_bandcomm;
    // int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (pSPARC->isGammaPoint) {
        if (!flagNoDmcomm) {
            // sum over spin comm group
            if(pSPARC->npspin > 1) {        
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos, nuChi0EigsAmounts*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
            }
            // sum over all k-point groups
            if (pSPARC->npkpt > 1) {            
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos, nuChi0EigsAmounts*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
            }
            // sum over all band groups 
            if (pSPARC->npband) {
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos, nuChi0EigsAmounts*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
            }
            if ((pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0)) {
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *outputDrhos = fopen("deltaRhos.txt", "w");
                    if (outputDrhos ==  NULL) {
                        printf("error printing delta rho s in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            for (int index = 0; index < DMnd; index++) {
                                fprintf(outputDrhos, "%12.9f\n", pRPA->deltaRhos[nuChi0EigIndex*DMnd + index]);
                            }
                            fprintf(outputDrhos, "\n");
                        }
                    }
                    fclose(outputDrhos);
                }
            }
        }
        for (int i = 0; i < nuChi0EigsAmounts; i++)
            transfer_deltaRho(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaRhos + i*DMnd, pRPA->deltaRhos_phi + i*pSPARC->Nd_d);

    } else {
        if (!flagNoDmcomm) {
            // sum over spin comm group
            if(pSPARC->npspin > 1) {        
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos_kpt, nuChi0EigsAmounts*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->spin_bridge_comm);        
            }
            // sum over all k-point groups
            if (pSPARC->npkpt > 1) {            
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos_kpt, nuChi0EigsAmounts*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->kpt_bridge_comm);
            }
            // sum over all band groups 
            if (pSPARC->npband) {
                MPI_Allreduce(MPI_IN_PLACE, pRPA->deltaRhos_kpt, nuChi0EigsAmounts*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->blacscomm);
            }
            if ((pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0)) {
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *outputDrhos = fopen("deltaRhos_kpt.txt", "w");
                    if (outputDrhos ==  NULL) {
                        printf("error printing delta rho s in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            for (int index = 0; index < DMnd; index++) {
                                fprintf(outputDrhos, "%12.9f %12.9f\n", creal(pRPA->deltaRhos_kpt[nuChi0EigIndex*DMnd + index]), cimag(pRPA->deltaRhos_kpt[nuChi0EigIndex*DMnd + index]));
                            }
                            fprintf(outputDrhos, "\n");
                        }
                    }
                    fclose(outputDrhos);
                }
            }
        }
        for (int i = 0; i < nuChi0EigsAmounts; i++)
            transfer_deltaRho_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaRhos_kpt + i*DMnd, pRPA->deltaRhos_kpt_phi + i*pSPARC->Nd_d);
    }
}

void Calculate_deltaRhoPotential(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex) {

}

void transfer_deltaRho(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *rho_send, double *rho_recv) {
    #ifdef DEBUG
    double t1, t2;
    #endif
        int rank;
        MPI_Comm_rank(nuChi0Eigscomm, &rank);
        
        int sdims[3], rdims[3], gridsizes[3];
        sdims[0] = pSPARC->npNdx; sdims[1] = pSPARC->npNdy; sdims[2] = pSPARC->npNdz;
        rdims[0] = pSPARC->npNdx_phi; rdims[1] = pSPARC->npNdy_phi; rdims[2] = pSPARC->npNdz_phi;
        gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        D2D(&pSPARC->d2d_dmcomm, &pSPARC->d2d_dmcomm_phi, gridsizes, pSPARC->DMVertices_dmcomm, rho_send, 
            pSPARC->DMVertices, rho_recv, (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, sdims, 
            pSPARC->dmcomm_phi, rdims, nuChi0Eigscomm, sizeof(double));
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, D2D took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
}

void transfer_deltaRho_kpt(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double _Complex *rho_send, double _Complex *rho_recv) {
    #ifdef DEBUG
    double t1, t2;
    #endif
        int rank;
        MPI_Comm_rank(nuChi0Eigscomm, &rank);
        
        int sdims[3], rdims[3], gridsizes[3];
        sdims[0] = pSPARC->npNdx; sdims[1] = pSPARC->npNdy; sdims[2] = pSPARC->npNdz;
        rdims[0] = pSPARC->npNdx_phi; rdims[1] = pSPARC->npNdy_phi; rdims[2] = pSPARC->npNdz_phi;
        gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        D2D(&pSPARC->d2d_dmcomm, &pSPARC->d2d_dmcomm_phi, gridsizes, pSPARC->DMVertices_dmcomm, rho_send, 
            pSPARC->DMVertices, rho_recv, (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, sdims, 
            pSPARC->dmcomm_phi, rdims, nuChi0Eigscomm, sizeof(double _Complex));
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, D2D took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
}

