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

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "electrostatics_RPA.h"
#include "linearSolvers.h"
#include "tools_RPA.h"

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


void Calculate_deltaRhoPotential(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex) {
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    int nuChi0EigsAmounts = pRPA->nNuChi0Eigscomm;

    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    } 
    if (pSPARC->isGammaPoint) {
        double *rhs = (double *)calloc(sizeof(double), nuChi0EigsAmounts * pSPARC->Nd_d);
        for (int i = 0; i < nuChi0EigsAmounts * pSPARC->Nd_d; i++) {
            rhs[i] = 4.0 * M_PI * pRPA->deltaRhos_phi[i]; // the minus sign is in function poisson_residual
        }
        Calculate_deltaRhoPotential_gamma(pSPARC, pRPA->deltaVs_phi, rhs, nuChi0EigsAmounts, nuChi0EigscommIndex);
        free(rhs);
        // printing delta V, but they are not the delta V for the next filtering
        for (int i = 0; i < nuChi0EigsAmounts; i++) {
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_phi + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm);
        }
        if ((pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0)) {
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *outputDVs = fopen("new_deltaVs.txt", "w");
                    if (outputDVs ==  NULL) {
                        printf("error printing new delta V s in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                                fprintf(outputDVs, "%12.9f\n", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                            }
                            fprintf(outputDVs, "\n");
                        }
                    }
                    fclose(outputDVs);
                }
            }
    } else {
        double _Complex *rhs = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts * pSPARC->Nd_d);
        for (int i = 0; i < nuChi0EigsAmounts * pSPARC->Nd_d; i++) {
            rhs[i] = 4.0 * M_PI * pRPA->deltaRhos_kpt_phi[i];
        }
        double qptx = pRPA->k1[qptIndex]; double qpty = pRPA->k2[qptIndex]; double qptz = pRPA->k3[qptIndex];
        Calculate_deltaRhoPotential_kpt(pSPARC, qptx, qpty, qptz, pRPA->deltaVs_kpt_phi, rhs, nuChi0EigsAmounts, nuChi0EigscommIndex);
        free(rhs);
    }
}

void Calculate_deltaRhoPotential_gamma(SPARC_OBJ *pSPARC, double *deltaVs_phi, double *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int Nd_d = pSPARC->Nd_d;
    // function pointer that applies b + Laplacian * x
    void (*residule_fptr) (SPARC_OBJ*, int, double, double*, double*, double*, MPI_Comm, double*) = poisson_residual; // poisson_residual is defined in lapVecRoutines.c
    void (*Jacobi_fptr) (SPARC_OBJ*, int, double, double*, double*, MPI_Comm) = Jacobi_preconditioner;
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
        double int_rhs = 0.0;
        // find integral of b, rho locally
        for (int i = 0; i < Nd_d; i++) {
            int_rhs += rhs[nuChi0EigsIndex*Nd_d + i];
        }
        int_rhs *= pSPARC->dV;
        double vt, vsum;
        vt = int_rhs;
        MPI_Allreduce(&vt, &vsum, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi); 
        int_rhs   = vsum;
        t2 = MPI_Wtime();
        if(!rank) 
            printf("nuChi0Eigscomm %d, dmcomm_phi rank %d, nuChi0EigsIndex %d, int_rhs = %18.14f, checking this took %.3f ms\n",
                nuChi0EigscommIndex, rank, nuChi0EigsIndex, int_rhs, (t2-t1)*1e3);
        t1 = MPI_Wtime();
    #endif    

        // call linear solver to solve the poisson equation
        // solve -Laplacian phi = 4 * M_PI * (rho + b) 
        // TODO: use real preconditioner instead of Jacobi preconditioner!
        if(pSPARC->POISSON_SOLVER == 0) {
            double omega, beta;
            int m, p;
            omega = 0.6, beta = 0.6; //omega = 0.6, beta = 0.6;
            m = 7, p = 6; //m = 9, p = 8; //m = 9, p = 9;
            AAR(pSPARC, residule_fptr, Jacobi_fptr, 0.0, Nd_d, deltaVs_phi + nuChi0EigsIndex*Nd_d, rhs + nuChi0EigsIndex*Nd_d, 
            omega, beta, m, p, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
        } else {
            if (rank == 0) printf("Please provide a valid poisson solver!\n");
            exit(EXIT_FAILURE);
            // CG(pSPARC, Lap_vec_mult, pSPARC->Nd, Nd_d, pSPARC->elecstPotential, rhs, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
        }

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif

        // shift the electrostatic potential so that its integral is zero for periodic systems
        if (pSPARC->BC == 2) {
            double deltaV_shift = 0.0;
            VectorSum  (deltaVs_phi + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm_phi);
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShift(deltaVs_phi + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm_phi);
        }
    }
}

void Calculate_deltaRhoPotential_kpt(SPARC_OBJ *pSPARC, double qptx, double qpty, double qptz, double _Complex *deltaVs_kpt_phi, double _Complex *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int Nd_d = pSPARC->Nd_d;
    int gammaFlag = 0;
    if (sqrt(qptx*qptx + qpty*qpty + qptz*qptz) < 1e-10) {
        gammaFlag = 1;
    }
    // function pointer that applies b + Laplacian * x
    void (*residule_fptr) (SPARC_OBJ*, int, double, double, double, double, double _Complex*, double _Complex*, double _Complex*, MPI_Comm, double*) = poisson_residual_kpt; // poisson_residual is defined in lapVecRoutines.c
    void (*Jacobi_fptr) (SPARC_OBJ*, int, double, double _Complex*, double _Complex*, MPI_Comm) = Jacobi_preconditioner_kpt;
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif    

        // call linear solver to solve the poisson equation
        // solve -Laplacian phi = 4 * M_PI * (rho + b) 
        // TODO: use real preconditioner instead of Jacobi preconditioner!
        if(pSPARC->POISSON_SOLVER == 0) {
            double omega, beta;
            int m, p;
            omega = 0.6, beta = 0.6; //omega = 0.6, beta = 0.6;
            m = 7, p = 6; //m = 9, p = 8; //m = 9, p = 9;
            AAR_kpt(pSPARC, residule_fptr, Jacobi_fptr, 0.0, Nd_d, qptx, qpty, qptz, deltaVs_kpt_phi + nuChi0EigsIndex*Nd_d, rhs + nuChi0EigsIndex*Nd_d, 
            omega, beta, m, p, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
        } else {
            if (rank == 0) printf("Please provide a valid poisson solver!\n");
            exit(EXIT_FAILURE);
            // CG(pSPARC, Lap_vec_mult, pSPARC->Nd, Nd_d, pSPARC->elecstPotential, rhs, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
        }

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif

        // shift the electrostatic potential so that its integral is zero for periodic systems
        if ((pSPARC->BC == 2) && gammaFlag) {
            double _Complex deltaV_shift = 0.0;
            VectorSumComplex  (deltaVs_kpt_phi + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm_phi);
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShiftComplex(deltaVs_kpt_phi + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm_phi);
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