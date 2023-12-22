#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "chebyshevFiltering.h"
#include "sternheimerEquation.h"
#include "electrostatics_RPA.h"
#include "tools_RPA.h"

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    int seed_offset = 0;
    double vec2norm = 0.0;
    if (pSPARC->isGammaPoint) {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                SeededRandVec(pRPA->deltaVs_phi + i*pSPARC->Nd_d, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm(pRPA->deltaVs_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScale(pRPA->deltaVs_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi); // unify the length of \Delta V
            }
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_phi + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print all \Delta V vectors of the last nuChi0Eigscomm.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                // only one processor print
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *output1stDV = fopen("deltaVs.txt", "w");
                    if (output1stDV ==  NULL) {
                        printf("error printing delta Vs in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            for (int index = 0; index < Nd; index++) {
                                fprintf(output1stDV, "%12.9f\n", pRPA->deltaVs[nuChi0EigIndex*Nd + index]);
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
        }
    }
    else {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                SeededRandVec_complex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm_complex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScaleComplex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi); // unify the length of \Delta V
            }
            Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pRPA->deltaVs_kpt + i * pSPARC->Nd_d_dmcomm);
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print the first \Delta V vector of the last nuChi0Eigscomm.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                // only one processor print
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *output1stDV = fopen("deltaVs_kpt.txt", "w");
                    if (output1stDV ==  NULL) {
                        printf("error printing delta Vs in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            for (int index = 0; index < Nd; index++) {
                                fprintf(output1stDV, "%12.9f %12.9f\n", creal(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd + index]), cimag(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd + index]));
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
        }
    }
}

void test_Hx_nuChi0(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (!flagNoDmcomm) {
        double *testHxAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm);
        test_Hx(pSPARC, testHxAccuracy);
        if ((nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->Nband_bandcomm > 0)) {
            printf("rank %d in nuChi0Eigscomm %d, the relative error epsilon*psi - H*psi of the bands %d %d are %.6E %.6E\n", rank, nuChi0EigscommIndex,
                   pSPARC->band_start_indx, pSPARC->band_start_indx + 1, testHxAccuracy[0], testHxAccuracy[1]);
        }
        free(testHxAccuracy);
    }

    if (nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) { // this test is done in only one nuChi0Eigscomm
        int printFlag = 1;
        int qptIndex = 1;
        int omegaIndex = 0;
        int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
        if (qptIndex > pRPA->Nqpts_sym - 1)
            qptIndex = pRPA->Nqpts_sym - 1;
        if (omegaIndex > pRPA->Nomega - 1)
            omegaIndex = pRPA->Nomega - 1;
        if (!flagNoDmcomm) {
            sternheimer_solver(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, printFlag);
        }
        collect_transfer_deltaRho(pSPARC, pRPA, nuChi0EigsAmount, printFlag);
        Calculate_deltaRhoPotential(pSPARC, pRPA, qptIndex, nuChi0EigsAmount, printFlag);
    }
}

void test_Hx(SPARC_OBJ *pSPARC, double *testHxAccuracy)
{ // make a test to see the accuracy of eigenvalues and eigenvectors saved in every nuChi0Eigscomm
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int *DMVertices = pSPARC->DMVertices_dmcomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    MPI_Comm comm = pSPARC->dmcomm;
    if (pSPARC->isGammaPoint) { // follow the sequence in function Calculate_elecDens, divide gamma-point, k-point first
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI, divide spin
            int sg = pSPARC->spin_start_indx + spn_i;
            Hamiltonian_vectors_mult(
                pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb + spn_i * DMnd, DMndsp, pSPARC->Yorb + spn_i * DMnd, DMndsp, spn_i, comm);
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * DMnd;
                double *Hpsi = pSPARC->Yorb + bandIndex * DMndsp + spn_i * DMnd;
                double eigValue = pSPARC->lambda[spn_i * ncol + bandIndex];
                for (int i = 0; i < DMnd; i++)
                {
                    testHxAccuracy[spn_i * ncol + bandIndex] += (*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i));
                }
            }
        }
    }
    else {
        int size_k = DMndsp * pSPARC->Nband_bandcomm;
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI_kpt
            int sg = pSPARC->spin_start_indx + spn_i;
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++) {
                Hamiltonian_vectors_mult_kpt(
                    pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                    pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb_kpt + kpt * size_k + spn_i * DMnd, DMndsp, pSPARC->Yorb_kpt + spn_i * DMnd, DMndsp, spn_i, kpt, comm);
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                    double _Complex *psi = pSPARC->Xorb_kpt + kpt * size_k + bandIndex * DMndsp + spn_i * DMnd;
                    double _Complex *Hpsi = pSPARC->Yorb_kpt + bandIndex * DMndsp + spn_i * DMnd;
                    double eigValue = pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    for (int i = 0; i < DMnd; i++) {
                        testHxAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex] += creal(conj(*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i)));
                    }
                }
            }
        }
    }
}

void sternheimer_solver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, int printFlag) // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
{
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    if (pSPARC->isGammaPoint) {   
        double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of 2-norm of residuals of all Sternheimer eq.s assigned in this processor
        for (int index = 0; index < pSPARC->Nd_d_dmcomm * nuChi0EigsAmount; index++) {
            pRPA->deltaRhos[index] = 0.0;
        }
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++) {
                double epsilon = pSPARC->lambda[spn_i * ncol + bandIndex];
                double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
                double bandWeight = pSPARC->occfac * pSPARC->occ[spn_i * ncol + bandIndex]; // occfac contains spin factor
                if (printFlag) {
                    char orbitalFileName[100];
                    snprintf(orbitalFileName, 100, "psi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputPsi = fopen(orbitalFileName, "w");
                    if (outputPsi ==  NULL) {
                        printf("error printing psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    } else {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                            fprintf(outputPsi, "%12.9f\n", psi[index]);
                        }
                    }
                    fclose(outputPsi);
                }
                sternSolverAccuracy[spn_i * ncol + bandIndex] = sternheimer_solver_gamma(pSPARC, spn_i, epsilon, pRPA->omega[omegaIndex], pRPA->deltaPsisReal, pRPA->deltaPsisImag, pRPA->deltaVs, psi, bandWeight, pRPA->deltaRhos, nuChi0EigsAmount);
                printf("spn_i %d, globalBandIndex %d, omegaIndex %d, stern res norm %.6E\n", spn_i, bandIndex + pSPARC->band_start_indx, omegaIndex, sternSolverAccuracy[spn_i * ncol + bandIndex]);
                // print the \Delta \psi vector from the first \Delta V.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                if (printFlag) {
                    char deltaOrbitalFileName[100];
                    snprintf(deltaOrbitalFileName, 100, "Dpsi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputDpsi = fopen(deltaOrbitalFileName, "w");
                    if (outputDpsi ==  NULL) {
                        printf("error printing delta psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                                fprintf(outputDpsi, "%12.9f %12.9f\n", pRPA->deltaPsisReal[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index], pRPA->deltaPsisImag[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                            }
                            fprintf(outputDpsi, "\n");
                        }
                    }
                    fclose(outputDpsi);
                }
            }
        }
        free(sternSolverAccuracy);
    }
    else {
        double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of norm of residuals of two Sternheimer eq.s
        for (int index = 0; index < pSPARC->Nd_d_dmcomm * nuChi0EigsAmount; index++) {
            pRPA->deltaRhos_kpt[index] = 0.0;
        }
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++) {
                int kg = kpt + pSPARC->kpt_start_indx;
                int kPq = pRPA->kPqList[kg][qptIndex];
                int kMq = pRPA->kMqList[kg][qptIndex];
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++) {
                    double epsilon = pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    double _Complex *psi_kpt = pSPARC->Xorb_kpt + kpt * ncol * DMndsp + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
                    double bandWeight = pSPARC->occfac * (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * pSPARC->occ[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    char orbitalFileName[100];
                    snprintf(orbitalFileName, 100, "psi_kpt%d_band%d_spin%d.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputPsi = fopen(orbitalFileName, "w");
                    if (outputPsi ==  NULL) {
                        printf("error printing psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    } else {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                            fprintf(outputPsi, "%12.9f %12.9f\n", creal(psi_kpt[index]), cimag(psi_kpt[index]));
                        }
                    }
                    fclose(outputPsi);
                    sternSolverAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex] = sternheimer_solver_kpt(pSPARC, spn_i, kPq, kMq, epsilon, 
                         pRPA->omega[omegaIndex], pRPA->deltaPsis_kpt, pRPA->deltaVs_kpt, psi_kpt, bandWeight, pRPA->deltaRhos_kpt, nuChi0EigsAmount); // solve the two Sternheimer eq.s (-i\omega and +i\omega) together
                    printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, +-omega, stern res norm %.6E\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, 
                         omegaIndex, sternSolverAccuracy[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex]);

                    char deltaOrbitalFileName[100];
                    snprintf(deltaOrbitalFileName, 100, "Dpsi_kpt%d_band%d_spin%d_-omega.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputDpsiMinus = fopen(deltaOrbitalFileName, "w");
                    snprintf(deltaOrbitalFileName, 100, "Dpsi_kpt%d_band%d_spin%d_+omega.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputDpsiPlus = fopen(deltaOrbitalFileName, "w");
                    if ((outputDpsiMinus ==  NULL) || (outputDpsiPlus ==  NULL)) {
                        printf("error printing delta psi kpt %d, band %d, spin %d\n", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    }
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) { // print \Delta psi for the last \Delta V
                        fprintf(outputDpsiMinus, "%12.9f %12.9f\n", creal(pRPA->deltaPsis_kpt[index]), cimag(pRPA->deltaPsis_kpt[index]));
                        fprintf(outputDpsiPlus, "%12.9f %12.9f\n", creal(pRPA->deltaPsis_kpt[pSPARC->Nd_d_dmcomm + index]), cimag(pRPA->deltaPsis_kpt[pSPARC->Nd_d_dmcomm + index]));
                    }
                    fclose(outputDpsiMinus);
                    fclose(outputDpsiPlus);
                }
            }
        }
        free(sternSolverAccuracy);
    }
    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    double minEig = 0.0;
    if (!nuChi0EigscommIndex) {
        minEig = find_min_eigenvalue(pSPARC, pRPA, qptIndex, omegaIndex, flagNoDmcomm);
    }
    MPI_Bcast(&minEig, 1, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm);
}

double find_min_eigenvalue(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm) { // find min eigenvalue by power method
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double minEig = 0.0;
    double vec2norm = 0.0;
    int loopFlag = 1;
    int maxIter = 30;
    int iter = 0;
    while (loopFlag) {
        if (pSPARC->isGammaPoint) {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm(pRPA->deltaVs_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScale(pRPA->deltaVs_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_phi, pRPA->deltaVs);
        } else {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm_complex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScaleComplex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
            Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_kpt_phi, pRPA->deltaVs_kpt);
        }
        if ((fabs(-vec2norm - minEig) < 2e-4) || (iter == maxIter) || (!pSPARC->isGammaPoint)) { // current kpt calculation is not available
            loopFlag = 0;
        }
        minEig = -vec2norm;
        if (!flagNoDmcomm) {
            sternheimer_solver(pSPARC, pRPA, qptIndex, omegaIndex, 1, 0);
        }
        collect_transfer_deltaRho(pSPARC, pRPA, 1, 0);
        Calculate_deltaRhoPotential(pSPARC, pRPA, qptIndex, 1, 0);
        iter++;
    }
    if (iter == maxIter) printf("qpt %d, omega %d, minimum eigenvalue does not reach the required accuracy.\n", qptIndex, omegaIndex);
    #ifdef DEBUG
    if (!rank) printf("qpt %d, omega %d, iterated for %d times, found minEig %.5E\n", qptIndex, omegaIndex, iter, minEig);
    #endif
    return minEig;
}