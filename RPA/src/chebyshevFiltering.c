#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "chebyshevFiltering.h"
#include "linearSolvers.h"

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
        double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm * 2);
        test_chi0_times_deltaV(pSPARC, pRPA, qptIndex, omegaIndex, sternSolverAccuracy);
        free(sternSolverAccuracy);
    }
}

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA)
{
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    int seed_offset = 0;
    if (pSPARC->isGammaPoint)
    {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++)
        {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL)
            {
                SeededRandVec(pRPA->initDeltaVs, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
            }
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->initDeltaVs, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
        }
    }
    else
    {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++)
        {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL)
            {
                SeededRandVec_complex(pRPA->initDeltaVs_kpt, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
            }
            Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->initDeltaVs_kpt, pRPA->deltaVs_kpt + i * pSPARC->Nd_d_dmcomm);
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

// void test_chi0_times_deltaV(SPARC_OBJ *pSPARC, int **kPqSymList, int qptIndex, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, int nuChi0EigsAmount, double *sternSolverAccuracy) {
void test_chi0_times_deltaV(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double *sternSolverAccuracy)
{
    int nuChi0EigsAmounts = 1;
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    if (pSPARC->isGammaPoint)
    {
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        {
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++)
            {
                double epsilon = pSPARC->lambda[spn_i * ncol + bandIndex];
                double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
                sternSolverAccuracy[spn_i * ncol + bandIndex] = test_chi0_times_deltaV_gamma(pSPARC, spn_i, epsilon, pRPA->omega[omegaIndex], pRPA->deltaPsisReal, pRPA->deltaPsisImag, pRPA->deltaVs, psi, nuChi0EigsAmounts);
                printf("spn_i %d, globalBandIndex %d, omegaIndex %d, stern res norm %.6E\n", spn_i, bandIndex + pSPARC->band_start_indx, omegaIndex, sternSolverAccuracy[spn_i * ncol + bandIndex]);
            }
        }
    }
    else
    {
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
                    sternSolverAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex] = test_chi0_times_deltaV_kpt(pSPARC, spn_i, kPq, kMq, epsilon, 
                         pRPA->omega[omegaIndex], pRPA->deltaPsis_kpt, pRPA->deltaVs_kpt, psi_kpt, nuChi0EigsAmounts);
                    printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, -omega, stern res norm %.6E\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, 
                         omegaIndex, sternSolverAccuracy[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex]);
                    sternSolverAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex + 1] = test_chi0_times_deltaV_kpt(pSPARC, spn_i, kPq, kMq, epsilon, 
                         -pRPA->omega[omegaIndex], pRPA->deltaPsis_kpt + pSPARC->Nd_d_dmcomm, pRPA->deltaVs_kpt, psi_kpt, nuChi0EigsAmounts);
                    printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, +omega, stern res norm %.6E\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, 
                         omegaIndex, sternSolverAccuracy[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex + 1]);
                }
            }
        }
    }
}

double test_chi0_times_deltaV_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, int nuChi0EigsAmounts)
{
    void (*lhsfun)(SPARC_OBJ *, int, double, double, double *, double *, double _Complex *, int) = Sternheimer_lhs;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double _Complex *SternheimerRhs = (double _Complex *)calloc(sizeof(double _Complex), DMnd);
    for (int i = 0; i < DMnd; i++)
    {
        SternheimerRhs[i] = -(double _Complex)(deltaVs[i] * psi[i]);
    }

    set_initial_guess_deltaPsis(pSPARC, spn_i, epsilon, omega, SternheimerRhs, deltaPsisReal, deltaPsisImag);

    double *resNormRecords = (double *)calloc(sizeof(double), 1000 * nuChi0EigsAmounts); // 1000 is maximum iteration time
    int iterTime = block_COCG(lhsfun, pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, SternheimerRhs, nuChi0EigsAmounts, 1000, resNormRecords);

    double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double residualNorm = 0.0;
    Sternheimer_lhs(pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, residual, nuChi0EigsAmounts);
    for (int i = 0; i < DMnd; i++)
    {
        residual[i] -= SternheimerRhs[i];
        residualNorm += conj(residual[i]) * residual[i];
    }
    free(SternheimerRhs);
    free(resNormRecords);
    free(residual);
    return residualNorm;
}

void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts)
{
    int sg = pSPARC->spin_start_indx + spn_i;
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    double *lhsXreal_Xcomp = (double*)calloc(sizeof(double), DMnd * nuChi0EigsAmounts);
    Hamiltonian_vectors_mult( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Xreal, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += Xreal[i];
    }
    Hamiltonian_vectors_mult( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Ximag, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += (Ximag[i] - omega) * I;
    }
    free(lhsXreal_Xcomp);
}

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag)
{
    // going to add the code for generating the initial guess based on psis
}

double test_chi0_times_deltaV_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi_kpt, int nuChi0EigsAmounts)
{
    void (*lhsfun)(SPARC_OBJ *, int, int, double, double, double _Complex *, double _Complex *, int) = Sternheimer_lhs_kpt;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double _Complex *SternheimerRhs_kpt = (double _Complex *)calloc(sizeof(double _Complex), DMnd);
    for (int i = 0; i < DMnd; i++)
    {
        SternheimerRhs_kpt[i] = -deltaVs_kpt[i] * psi_kpt[i];
    }

    set_initial_guess_deltaPsis_kpt(pSPARC, spn_i, kPq, kMq, epsilon, omega, SternheimerRhs_kpt, deltaPsis_kpt);

    double *resNormRecords = (double *)calloc(sizeof(double), 1000 * nuChi0EigsAmounts); // 1000 is maximum iteration time
    int iterTime = kpt_solver(lhsfun, pSPARC, spn_i, kPq, epsilon, omega, deltaPsis_kpt, SternheimerRhs_kpt, nuChi0EigsAmounts, 1000, resNormRecords);

    double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd);
    double residualNorm = 0.0;
    Sternheimer_lhs_kpt(pSPARC, spn_i, kPq, epsilon, omega, deltaPsis_kpt, residual, 1);
    for (int i = 0; i < DMnd; i++)
    {
        residual[i] -= SternheimerRhs_kpt[i];
        residualNorm += conj(residual[i]) * residual[i];
    }
    free(SternheimerRhs_kpt);
    free(resNormRecords);
    free(residual);
    return residualNorm;
}

void set_initial_guess_deltaPsis_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *SternheimerRhs, double _Complex *deltaPsis)
{
    // going to add the code for generating the initial guess based on psis
}

void Sternheimer_lhs_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *X, double _Complex *lhsX, int nuChi0EigsAmounts)
{
    int sg = pSPARC->spin_start_indx + spn_i;
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Hamiltonian_vectors_mult_kpt( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
        pSPARC, DMnd, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, X, DMnd, lhsX, DMnd, spn_i, kPq, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd; i++) // different from gamma point case, Sternheimer eq.s for different \Delta Vs are solved one by one
    {
        lhsX[i] -= omega * I;
    }
}