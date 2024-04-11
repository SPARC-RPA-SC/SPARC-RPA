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
#include "lapVecRoutines.h"
#include "exactExchange.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "electrostatics_RPA.h"
#include "linearSolvers.h"
#include "tools_RPA.h"

void collect_deltaRho_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm) {
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int DMnd = pSPARC->Nd_d_dmcomm;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (!flagNoDmcomm) {
        // sum over spin comm group
        if(pSPARC->npspin > 1) {        
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
        }
        // sum over all k-point groups
        if (pSPARC->npkpt > 1) {            
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
        // sum over all band groups 
        if (pSPARC->npband) {
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && printFlag) {
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                FILE *outputDrhos = fopen("deltaRhos.txt", "w");
                if (outputDrhos ==  NULL) {
                    printf("error printing delta rho s in test\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                        for (int index = 0; index < DMnd; index++) {
                            fprintf(outputDrhos, "%12.9f\n", deltaRhos[nuChi0EigIndex*DMnd + index]);
                        }
                        fprintf(outputDrhos, "\n");
                    }
                }
                fclose(outputDrhos);
            }
        }
    }
}

void collect_deltaRho_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm) {
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int ncol = pSPARC->Nband_bandcomm;
    // int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (!flagNoDmcomm) {
        // sum over spin comm group
        if(pSPARC->npspin > 1) {        
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->spin_bridge_comm);        
        }
        // sum over all k-point groups
        if (pSPARC->npkpt > 1) {            
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
        // sum over all band groups 
        if (pSPARC->npband) {
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->blacscomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && printFlag) {
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                FILE *outputDrhos = fopen("deltaRhos_kpt.txt", "w");
                if (outputDrhos ==  NULL) {
                    printf("error printing delta rho s in test\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                        for (int index = 0; index < DMnd; index++) {
                            fprintf(outputDrhos, "%12.9f %12.9f\n", creal(deltaRhos_kpt[nuChi0EigIndex*DMnd + index]), cimag(deltaRhos_kpt[nuChi0EigIndex*DMnd + index]));
                        }
                        fprintf(outputDrhos, "\n");
                    }
                }
                fclose(outputDrhos);
            }
        }
    }
}


void Calculate_sqrtNu_vecs_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, double *deltaVs, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    if (flagNoDmcomm) {
        return;
    }
    poissonSolver_gamma(pSPARC, deltaVs, deltaRhos, nuChi0EigsAmount, nuChi0EigscommIndex);
    if (printFlag) {
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                FILE *outputDVs = fopen("new_deltaVs.txt", "w");
                if (outputDVs ==  NULL) {
                    printf("error printing new delta V s in test\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                            fprintf(outputDVs, "%12.9f\n", deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputDVs, "\n");
                    }
                }
                fclose(outputDVs);
            }
        }
    }
}

void Calculate_deltaRhoPotential_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, double _Complex *deltaVs_kpt, double qptx, double qpty, double qptz, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (flagNoDmcomm) {
        return;
    }

    double _Complex *rhs = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmount * pSPARC->Nd_d_dmcomm);
    for (int i = 0; i < nuChi0EigsAmount * pSPARC->Nd_d_dmcomm; i++) {
        rhs[i] = 4.0 * M_PI * deltaRhos_kpt[i];
    }
    poissonSolver_kpt(pSPARC, qptx, qpty, qptz, deltaVs_kpt, rhs, nuChi0EigsAmount, nuChi0EigscommIndex);
    free(rhs);
}

void poissonSolver_gamma(SPARC_OBJ *pSPARC, double *deltaVs, double *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex) {
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int Nd_d = pSPARC->Nd_d_dmcomm;

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif
        double deltaV_shift = 0.0;
        VectorSum (rhs + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm);
        if (pSPARC->BC == 2) {
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShift(rhs + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!globalRank) 
            printf("nuChi0Eigscomm %d, global rank %d, nuChi0EigsIndex %d, initial int_rhs = %18.14f, checking this took %.3f ms\n",
                nuChi0EigscommIndex, globalRank, nuChi0EigsIndex, deltaV_shift, (t2-t1)*1e3);
        t1 = MPI_Wtime();
    #endif    
        // no need to multiply 4pi on the rhs, it is included in pois_const 
        pois_kron(pSPARC, rhs + nuChi0EigsIndex*Nd_d, pSPARC->pois_const, 1, deltaVs + nuChi0EigsIndex*Nd_d);
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (globalRank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif

        // shift the electrostatic potential so that its integral is zero for periodic systems
        if (pSPARC->BC == 2) {
            deltaV_shift = 0.0;
            VectorSum  (deltaVs + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm);
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShift(deltaVs + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm);
        }
    }
}

void poissonSolver_kpt(SPARC_OBJ *pSPARC, double qptx, double qpty, double qptz, double _Complex *deltaVs_kpt, double _Complex *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int Nd_d = pSPARC->Nd_d_dmcomm;
    int gammaFlag = 0;
    if (sqrt(qptx*qptx + qpty*qpty + qptz*qptz) < 1e-10) {
        gammaFlag = 1;
    }

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif    

        // call linear solver to solve the poisson equation
        // solve -Laplacian phi = 4 * M_PI * (rho + b) 
        // TODO: use real preconditioner instead of Jacobi preconditioner!

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif

        // shift the electrostatic potential so that its integral is zero for periodic systems
        if ((pSPARC->BC == 2) && gammaFlag) {
            double _Complex deltaV_shift = 0.0;
            VectorSumComplex  (deltaVs_kpt + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm);
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShiftComplex(deltaVs_kpt + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm);
        }
    }
}

void poisson_residual_kpt(SPARC_OBJ *pSPARC, int N, double qptx, double qpty, double qptz, double c, double _Complex *x, double _Complex *b, double _Complex *r, MPI_Comm comm, double *time_info) 
{
    // int i;
    // double t1 = MPI_Wtime();
    // Lap_vec_mult_kpt(pSPARC, N, pSPARC->DMVertices, 1, 0.0, x, N, r, N, kpt, comm); // it needs to be modified to receive coords of the q-point
    // double t2 = MPI_Wtime();
    // *time_info = t2 - t1;
    
    // // Calculate residual once Lx is obtained
    // for (i = 0; i < N; i++) r[i] += b[i];
}

void Jacobi_preconditioner_kpt(SPARC_OBJ *pSPARC, int N, double c, double _Complex*r, double _Complex*f, MPI_Comm comm) {

}