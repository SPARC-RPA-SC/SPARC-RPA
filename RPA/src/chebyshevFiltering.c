#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "chebyshevFiltering.h"
#include "sternheimerEquation.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA)
{
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    initialize_deltaVs(pSPARC, pRPA);
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (flagNoDmcomm)
        return;

    double *testHxAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm);
    test_Hx(pSPARC, testHxAccuracy);
    if ((nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->Nband_bandcomm > 0))
    {
        printf("rank %d in nuChi0Eigscomm %d, the relative error epsilon*psi - H*psi of the bands %d %d are %.6E %.6E\n", rank, nuChi0EigscommIndex,
               pSPARC->band_start_indx, pSPARC->band_start_indx + 1, testHxAccuracy[0], testHxAccuracy[1]);
    }
    free(testHxAccuracy);

    if ((nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->dmcomm != MPI_COMM_NULL))
    { // this test is done in only one nuChi0Eigscomm
        int qptIndex = 1;
        int omegaIndex = 0;
        if (qptIndex > pRPA->Nqpts_sym - 1)
            qptIndex = pRPA->Nqpts_sym - 1;
        if (omegaIndex > pRPA->Nomega - 1)
            omegaIndex = pRPA->Nomega - 1;
        test_sternheimer_solver(pSPARC, pRPA, qptIndex, omegaIndex);
    }
}

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA)
{
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    int seed_offset = 0;
    if (pSPARC->isGammaPoint) {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                SeededRandVec(pRPA->initDeltaVs, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
            }
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->initDeltaVs, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print the first \Delta V vector of the last nuChi0Eigscomm.
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
                SeededRandVec_complex(pRPA->initDeltaVs_kpt, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
            }
            Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->initDeltaVs_kpt, pRPA->deltaVs_kpt + i * pSPARC->Nd_d_dmcomm);
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

void test_Hx(SPARC_OBJ *pSPARC, double *testHxAccuracy)
{ // make a test to see the accuracy of eigenvalues and eigenvectors saved in every nuChi0Eigscomm
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int *DMVertices = pSPARC->DMVertices_dmcomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    MPI_Comm comm = pSPARC->dmcomm;
    if (pSPARC->isGammaPoint)
    { // follow the sequence in function Calculate_elecDens, divide gamma-point, k-point first
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        { // follow the sequence in function eigSolve_CheFSI, divide spin
            int sg = pSPARC->spin_start_indx + spn_i;
            Hamiltonian_vectors_mult(
                pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb + spn_i * DMnd, DMndsp, pSPARC->Yorb + spn_i * DMnd, DMndsp, spn_i, comm);
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++)
            { // verify the correctness of psi
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
    else
    {
        int size_k = DMndsp * pSPARC->Nband_bandcomm;
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        { // follow the sequence in function eigSolve_CheFSI_kpt
            int sg = pSPARC->spin_start_indx + spn_i;
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++)
            {
                Hamiltonian_vectors_mult_kpt(
                    pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                    pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb_kpt + kpt * size_k + spn_i * DMnd, DMndsp, pSPARC->Yorb_kpt + spn_i * DMnd, DMndsp, spn_i, kpt, comm);
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++)
                { // verify the correctness of psi
                    double _Complex *psi = pSPARC->Xorb_kpt + kpt * size_k + bandIndex * DMndsp + spn_i * DMnd;
                    double _Complex *Hpsi = pSPARC->Yorb_kpt + bandIndex * DMndsp + spn_i * DMnd;
                    double eigValue = pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    for (int i = 0; i < DMnd; i++)
                    {
                        testHxAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex] += creal(conj(*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i)));
                    }
                }
            }
        }
    }
}

// void test_sternheimer_solver(SPARC_OBJ *pSPARC, int **kPqSymList, int qptIndex, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, int nuChi0EigsAmount, double *sternSolverAccuracy) {
void test_sternheimer_solver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex)
{
    int nuChi0EigsAmounts = pRPA->nNuChi0Eigscomm; // only the first \Delta V in the nuChi0Eigscomm takes part in the test
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    if (pSPARC->isGammaPoint)
    {   double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of 2-norm of residuals of all Sternheimer eq.s assigned in this processor
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        {
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++)
            {
                double epsilon = pSPARC->lambda[spn_i * ncol + bandIndex];
                double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
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
                sternSolverAccuracy[spn_i * ncol + bandIndex] = sternheimer_solver_gamma(pSPARC, spn_i, epsilon, pRPA->omega[omegaIndex], pRPA->deltaPsisReal, pRPA->deltaPsisImag, pRPA->deltaVs, psi, nuChi0EigsAmounts);
                printf("spn_i %d, globalBandIndex %d, omegaIndex %d, stern res norm %.6E\n", spn_i, bandIndex + pSPARC->band_start_indx, omegaIndex, sternSolverAccuracy[spn_i * ncol + bandIndex]);
                // print the \Delta \psi vector from the first \Delta V.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                char deltaOrbitalFileName[100];
                snprintf(deltaOrbitalFileName, 100, "Dpsi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                FILE *outputDpsi = fopen(deltaOrbitalFileName, "w");
                if (outputDpsi ==  NULL) {
                    printf("error printing delta psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    exit(EXIT_FAILURE);
                } else {
                    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmounts; nuChi0EigIndex++) {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                            fprintf(outputDpsi, "%12.9f %12.9f\n", pRPA->deltaPsisReal[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index], pRPA->deltaPsisImag[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputDpsi, "\n");
                    }
                }
                fclose(outputDpsi);
            }
        }
        free(sternSolverAccuracy);
    }
    else
    {
        double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm * 2);
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        {
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++)
            {
                int kg = kpt + pSPARC->kpt_start_indx;
                int kPq = pRPA->kPqList[kg][qptIndex];
                int kMq = pRPA->kMqList[kg][qptIndex];
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++)
                {
                    double epsilon = pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    double _Complex *psi_kpt = pSPARC->Xorb_kpt + kpt * ncol * DMndsp + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
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
                         pRPA->omega[omegaIndex], pRPA->deltaPsis_kpt, pRPA->deltaVs_kpt, psi_kpt, nuChi0EigsAmounts);
                    printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, -omega, stern res norm %.6E\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, 
                         omegaIndex, sternSolverAccuracy[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex]);
                    char deltaOrbitalFileName[100];
                    snprintf(deltaOrbitalFileName, 100, "Dpsi_kpt%d_band%d_spin%d_-omega.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    FILE *outputDpsi = fopen(deltaOrbitalFileName, "w");
                    if (outputDpsi ==  NULL) {
                        printf("error printing delta psi kpt %d, band %d, spin %d -\n", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    } else {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) { // print \Delta psi for the last \Delta V
                            fprintf(outputDpsi, "%12.9f %12.9f\n", creal(pRPA->deltaPsis_kpt[index]), cimag(pRPA->deltaPsis_kpt[index]));
                        }
                    }
                    fclose(outputDpsi);
                    sternSolverAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex + 1] = sternheimer_solver_kpt(pSPARC, spn_i, kPq, kMq, epsilon, 
                         -pRPA->omega[omegaIndex], pRPA->deltaPsis_kpt + pSPARC->Nd_d_dmcomm, pRPA->deltaVs_kpt, psi_kpt, nuChi0EigsAmounts);
                    printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, +omega, stern res norm %.6E\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, 
                         omegaIndex, sternSolverAccuracy[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex + 1]);
                    snprintf(deltaOrbitalFileName, 100, "Dpsi_kpt%d_band%d_spin%d_+omega.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                    outputDpsi = fopen(deltaOrbitalFileName, "w");
                    if (outputDpsi ==  NULL) {
                        printf("error printing delta psi kpt %d, band %d, spin %d +\n", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                        exit(EXIT_FAILURE);
                    } else {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) { // print \Delta psi for the last \Delta V
                            fprintf(outputDpsi, "%12.9f %12.9f\n", creal(pRPA->deltaPsis_kpt[index]), cimag(pRPA->deltaPsis_kpt[index]));
                        }
                    }
                    fclose(outputDpsi);
                }
            }
        }
        free(sternSolverAccuracy);
    }
}



