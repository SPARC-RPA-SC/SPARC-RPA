#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "linearSolvers.h"
#include "sternheimerEquation.h"

void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, int printFlag) // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
{
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;

    double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of 2-norm of residuals of all Sternheimer eq.s assigned in this processor
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

    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}

void sternheimer_eq_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, int printFlag) { // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;

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

    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}

double sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, double bandWeight, double *deltaRhos, int nuChi0EigsAmounts)
{
    void (*lhsfun)(SPARC_OBJ *, int, double, double, double *, double *, double _Complex *, int) = Sternheimer_lhs;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs[nuChi0EigsIndex*DMnd + i] = -deltaVs[nuChi0EigsIndex*DMnd + i]*(psi[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
            deltaPsisReal[nuChi0EigsIndex*DMnd + i] = 0.0;
            deltaPsisImag[nuChi0EigsIndex*DMnd + i] = 0.0;
        }
    }

    set_initial_guess_deltaPsis(pSPARC, spn_i, epsilon, omega, SternheimerRhs, deltaPsisReal, deltaPsisImag);

    int maxIter = 1000;
    double *resNormRecords = (double *)calloc(sizeof(double), maxIter * nuChi0EigsAmounts); // 1000 is maximum iteration time
    int iterTime = block_COCG(lhsfun, pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, SternheimerRhs, nuChi0EigsAmounts, 1e-8, maxIter, resNormRecords);

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) { // bandWeight includes occupation and spin factor
            deltaRhos[nuChi0EigsIndex * DMnd + i] += 2 * bandWeight * deltaPsisReal[nuChi0EigsIndex * DMnd + i] * (psi[i] / sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd * nuChi0EigsAmounts);
    double residualNorm = 0.0;
    Sternheimer_lhs(pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, residual, nuChi0EigsAmounts);
    for (int i = 0; i < nuChi0EigsAmounts*DMnd; i++) { // the sum of square of residual norms of all Sternheimer eq.s of the \psi
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
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {
        lhsX[i] = 0.0;
    }
    double *lhsXreal_Xcomp = (double*)calloc(sizeof(double), DMnd * nuChi0EigsAmounts);
    Hamiltonian_vectors_mult( // (Hamiltonian - \epsilon * I)
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Xreal, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += lhsXreal_Xcomp[i] - omega*Xreal[i] * I;
    }
    Hamiltonian_vectors_mult( // (Hamiltonian - \epsilon * I)
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Ximag, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += lhsXreal_Xcomp[i] * I + omega*Ximag[i];
    }
    free(lhsXreal_Xcomp);
}

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag)
{
    // going to add the code for generating the initial guess based on psis
}

double sternheimer_solver_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi_kpt, double bandWeight, double _Complex *deltaRhos_kpt, int nuChi0EigsAmounts)
{
    void (*lhsfun)(SPARC_OBJ *, int, int, double, double, double _Complex *, double _Complex *, int) = Sternheimer_lhs_kpt;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs_kpt = (double _Complex *)calloc(sizeof(double _Complex), DMnd);
    double residualNorm = 0.0;
    double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) { // solve each Sternheimer equation one by one, unlike gamma-point case
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs_kpt[i] = -deltaVs_kpt[nuChi0EigsIndex*DMnd + i]*(psi_kpt[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
        set_initial_guess_deltaPsis_kpt(pSPARC, spn_i, kPq, kMq, epsilon, omega, SternheimerRhs_kpt, deltaPsis_kpt);

        int maxIter = 1000;
        double *resNormRecords = (double *)calloc(sizeof(double), maxIter * nuChi0EigsAmounts); // 1000 is maximum iteration time
        int iterTime = kpt_solver(lhsfun, pSPARC, spn_i, kPq, epsilon, omega, deltaPsis_kpt, SternheimerRhs_kpt, nuChi0EigsAmounts, maxIter, resNormRecords);

        for (int i = 0; i < DMnd; i++) { // bandWeight includes occupation, kpt weight and spin factor
            deltaRhos_kpt[nuChi0EigsIndex * DMnd + i] += bandWeight * conj(psi_kpt[i] / sqrtdV) * (deltaPsis_kpt[i] + deltaPsis_kpt[DMnd + i]); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }

        Sternheimer_lhs_kpt(pSPARC, spn_i, kPq, epsilon, omega, deltaPsis_kpt, residual, 1);
        free(resNormRecords);
        for (int i = 0; i < DMnd; i++) { // the sum of square of residual norms of 1st Sternheimer eq. (-\i omega) of the \psi
            residual[i] -= SternheimerRhs_kpt[i];
            residualNorm += conj(residual[i]) * residual[i];
        }
        Sternheimer_lhs_kpt(pSPARC, spn_i, kPq, epsilon, -omega, deltaPsis_kpt + DMnd, residual, 1);
        for (int i = 0; i < DMnd; i++) { // the sum of square of residual norms of 2nd Sternheimer eq. (+\i omega) of the \psi
            residual[i] -= SternheimerRhs_kpt[i];
            residualNorm += conj(residual[i]) * residual[i];
        }
    }

    free(SternheimerRhs_kpt);
    free(residual);
    return residualNorm;
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

void set_initial_guess_deltaPsis_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *SternheimerRhs, double _Complex *deltaPsis)
{
    // going to add the code for generating the initial guess based on psis
}