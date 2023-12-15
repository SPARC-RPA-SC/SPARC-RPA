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

double sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, int nuChi0EigsAmounts)
{
    void (*lhsfun)(SPARC_OBJ *, int, double, double, double *, double *, double _Complex *, int) = Sternheimer_lhs;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs[nuChi0EigsIndex*DMnd + i] = -deltaVs[nuChi0EigsIndex*DMnd + i]*(psi[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    set_initial_guess_deltaPsis(pSPARC, spn_i, epsilon, omega, SternheimerRhs, deltaPsisReal, deltaPsisImag);

    int maxIter = 1000;
    double *resNormRecords = (double *)calloc(sizeof(double), maxIter * nuChi0EigsAmounts); // 1000 is maximum iteration time
    int iterTime = block_COCG(lhsfun, pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, SternheimerRhs, nuChi0EigsAmounts, 1e-8, maxIter, resNormRecords);

    double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
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
    Hamiltonian_vectors_mult( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Xreal, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += lhsXreal_Xcomp[i] - omega*Xreal[i] * I;
    }
    Hamiltonian_vectors_mult( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
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

double sternheimer_solver_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi_kpt, int nuChi0EigsAmounts)
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

        Sternheimer_lhs_kpt(pSPARC, spn_i, kPq, epsilon, omega, deltaPsis_kpt, residual, 1);
        free(resNormRecords);
        for (int i = 0; i < DMnd; i++) { // the sum of square of residual norms of all Sternheimer eq.s of the \psi
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