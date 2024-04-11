#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "linearSolvers.h"
#include "sternheimerEquation.h"

void collect_allXorb_allLambdas_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    MPI_Comm blacscomm = pSPARC->blacscomm;
    int blacscommRank, blacscommSize;
    MPI_Comm_rank(blacscomm, &blacscommRank);
    MPI_Comm_size(blacscomm, &blacscommSize);
    int *bandNumbers = (int *)calloc(sizeof(int), pSPARC->npband);
    int *bandStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    MPI_Allgather(&pSPARC->Nband_bandcomm, 1, MPI_INT, bandNumbers, 1, MPI_INT, blacscomm);
    MPI_Allgather(&pSPARC->band_start_indx, 1, MPI_INT, bandStartIndices, 1, MPI_INT, blacscomm);
    int *XorbLengths = (int *)calloc(sizeof(int), pSPARC->npband);
    int *XorbStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    for (int i = 0; i < pSPARC->npband; i++) {
        XorbLengths[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandNumbers[i];
        XorbStartIndices[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandStartIndices[i];
    }
    // collect all Xorb
    int localXorbLength = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;
    int allXorbLength = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates;
    MPI_Allgatherv(&pSPARC->Xorb[0], localXorbLength, MPI_DOUBLE, &pRPA->allXorb[0], XorbLengths, XorbStartIndices, MPI_DOUBLE, blacscomm);
    if (pSPARC->Nspin_spincomm > 1) { // modify the sequence of Xorbs to collect psis with the same spin
        double *temporaryAllXorb = (double *)calloc(sizeof(double), allXorbLength);
        memcpy(temporaryAllXorb, pRPA->allXorb, sizeof(double) * allXorbLength);
        for (int band = 0; band < pSPARC->Nstates; band++) {
            memcpy(&pRPA->allXorb[band * pSPARC->Nd_d_dmcomm], &temporaryAllXorb[band * pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm], sizeof(double) * pSPARC->Nd_d_dmcomm);
            memcpy(&pRPA->allXorb[pSPARC->Nd_d_dmcomm * pSPARC->Nstates + band * pSPARC->Nd_d_dmcomm], &temporaryAllXorb[(band * pSPARC->Nspin_spincomm + 1) * pSPARC->Nd_d_dmcomm], sizeof(double) * pSPARC->Nd_d_dmcomm);
        }
        free(temporaryAllXorb);
    }
    // collect all lambdas
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        MPI_Allgatherv(&pSPARC->lambda[spin * pSPARC->Nband_bandcomm], pSPARC->Nband_bandcomm, MPI_DOUBLE, &pRPA->allLambdas[spin * pSPARC->Nstates], bandNumbers, bandStartIndices, MPI_DOUBLE, blacscomm);
    }
    // if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (blacscommRank == blacscommSize - 1)) { // for checking the correctness
    //     char collectFileName[100];
    //     snprintf(collectFileName, 100, "nuChi0Eigscomm%d_blacsRank%d.Xorb", pRPA->nuChi0EigscommIndex, blacscommRank);
    //     FILE *outputXorb = fopen(collectFileName, "w");
    //     int testIndex = 2;
    //     for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
    //         for (int band = 0; band < pSPARC->Nstates; band++) {
    //             fprintf(outputXorb, "%d entry in spin %d band %d is %12.9f\n", testIndex, spin, band, pRPA->allXorb[spin * pSPARC->Nd_d_dmcomm * pSPARC->Nstates + band * pSPARC->Nd_d_dmcomm + testIndex]);
    //         }
    //     }
    //     fclose(outputXorb);
    // }
    free(bandNumbers);
    free(bandStartIndices);
    free(XorbLengths);
    free(XorbStartIndices);
}

void collect_allXorb_allLambdas_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex) {
}

void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, double *DVs, int printFlag) // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
{
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int ncol = pSPARC->Nband_bandcomm;

    // double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of 2-norm of residuals of all Sternheimer eq.s assigned in this processor
    for (int index = 0; index < pSPARC->Nd_d_dmcomm * nuChi0EigsAmount; index++) {
        pRPA->deltaRhos[index] = 0.0;
    }
    if (nuChi0EigsAmount < 1) return; // it is possible that some nuChi0Eigscomms are not assigned eigenpairs. For example: distribute 1500 into 120 nuChi0Eigscomm, then the last 4 comms will be left blank

    for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        for (int bandIndex = 0; bandIndex < ncol; bandIndex++) {
            if (pSPARC->occ[spn_i * ncol + bandIndex] < 1e-4) continue; // do not compute conduct band (blank band)
            double epsilon = pSPARC->lambda[spn_i * ncol + bandIndex];
            double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
            double bandWeight = pSPARC->occfac * pSPARC->occ[spn_i * ncol + bandIndex]; // occfac contains spin factor
            // if (printFlag) {
            //     char orbitalFileName[100];
            //     snprintf(orbitalFileName, 100, "psi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //     FILE *outputPsi = fopen(orbitalFileName, "w");
            //     if (outputPsi ==  NULL) {
            //         printf("error printing psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //         exit(EXIT_FAILURE);
            //     } else {
            //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            //             fprintf(outputPsi, "%12.9f\n", psi[index]);
            //         }
            //     }
            //     fclose(outputPsi);
            // }
            sternheimer_solver_gamma(pSPARC, spn_i, epsilon, pRPA->omega[omegaIndex], pRPA->flagPQ, pRPA->flagCOCGinitial, pRPA->allXorb, pRPA->allLambdas,
                pRPA->deltaPsisReal, pRPA->deltaPsisImag, DVs, psi, bandWeight, pRPA->deltaRhos, nuChi0EigsAmount,
                pRPA->sternRelativeResTol, pRPA->SternBlockSize[omegaIndex], printFlag, pRPA->timeRecordname); // sternSolverAccuracy[spn_i * ncol + bandIndex] = 
            // printf("spn_i %d, globalBandIndex %d, omegaIndex %d, stern res norm %.6E\n", spn_i, bandIndex + pSPARC->band_start_indx, omegaIndex, sternSolverAccuracy[spn_i * ncol + bandIndex]);
            
            // print the \Delta \psi vector from the first \Delta V.
            // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
            // if (printFlag) {
            //     char deltaOrbitalFileName[100];
            //     snprintf(deltaOrbitalFileName, 100, "Dpsi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //     FILE *outputDpsi = fopen(deltaOrbitalFileName, "w");
            //     if (outputDpsi ==  NULL) {
            //         printf("error printing delta psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //         exit(EXIT_FAILURE);
            //     } else {
            //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            //             for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
            //                 fprintf(outputDpsi, "%12.9f %12.9f  ", pRPA->deltaPsisReal[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index], pRPA->deltaPsisImag[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
            //             }
            //             fprintf(outputDpsi, "\n");
            //         }
            //     }
            //     fclose(outputDpsi);
            // }
        }
    }
    // free(sternSolverAccuracy);

    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}

void sternheimer_eq_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, double _Complex *DVs, int printFlag) { // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
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

void sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, int flagCOCGinitial, double *allXorb, double *allLambdas,
                                double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, double bandWeight, double *deltaRhos, int nuChi0EigsAmounts,
                                double sternRelativeResTol, int inputSternBlockSize, int printFlag, char *timeRecordname)
{
    void (*lhsfun)(SPARC_OBJ *, int, double *, double, double, int, double *, double *, double _Complex *, int) = Sternheimer_lhs;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs[nuChi0EigsIndex*DMnd + i] = -deltaVs[nuChi0EigsIndex*DMnd + i]*(psi[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
            deltaPsisReal[nuChi0EigsIndex*DMnd + i] = 0.0;
            deltaPsisImag[nuChi0EigsIndex*DMnd + i] = 0.0;
        }
        // linear operator P, only contains the term of psi to be computed, occ2 = occ
        if (flagPQ) {
            double _Complex dotProdPsiDeltaVPsi = 0.0;
            for (int i = 0; i < DMnd; i++) {
                dotProdPsiDeltaVPsi += -(psi[i])*SternheimerRhs[nuChi0EigsIndex*DMnd + i]; // don't add (psi[i]/sqrtdV)!
            }
            for (int i = 0; i < DMnd; i++) {
                SternheimerRhs[nuChi0EigsIndex*DMnd + i] += (psi[i])*dotProdPsiDeltaVPsi;
            }
        }
    }

    set_initial_guess_deltaPsis(pSPARC, spn_i, epsilon, omega, flagPQ, flagCOCGinitial, allXorb, allLambdas, SternheimerRhs, deltaPsisReal, deltaPsisImag, nuChi0EigsAmounts);

    // select the optimal block size
    int SternBlockSize = 1, blockSize = 0;
    int trySizeFlag = 1;
    double previousTrialTime = 0.0; double currentTrialTime = 0.0;
    int startIndex = 0;
    int nuChi0EigsRemainAmounts = nuChi0EigsAmounts;
    double sumLhsfunTime = 0.0, sumSolveMuTime = 0.0, sumMultipleTime = 0.0;
    FILE *outputFile;
    #ifdef DEBUG
    if (printFlag) outputFile = fopen(timeRecordname, "a");
    #endif
    double t5, t6;
    t5 = MPI_Wtime();
    while (trySizeFlag) {
        if (SternBlockSize > 1) {
            previousTrialTime = currentTrialTime;
        }
        double t1 = MPI_Wtime();
        blockSize = nuChi0EigsRemainAmounts > SternBlockSize ? SternBlockSize : nuChi0EigsRemainAmounts;
        int maxIter = pSPARC->Nd_d_dmcomm / blockSize;
        double *deltaPsisRealLoop = deltaPsisReal + startIndex*DMnd;
        double *deltaPsisImagLoop = deltaPsisImag + startIndex*DMnd;
        double _Complex *SternheimerRhsLoop = SternheimerRhs + startIndex*DMnd;
        double *resNormRecords = (double *)calloc(sizeof(double), (maxIter + 1) * blockSize); // 1000 is maximum iteration time
        double lhsfunTime = 0.0; double solveMuTime = 0.0; double multipleTime = 0.0;
        int iterTime = block_COCG(lhsfun, pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisRealLoop, deltaPsisImagLoop, SternheimerRhsLoop, blockSize, 
            sternRelativeResTol, maxIter, resNormRecords, &lhsfunTime, &solveMuTime, &multipleTime);
        free(resNormRecords);
        nuChi0EigsRemainAmounts -= blockSize;
        startIndex += blockSize;
        double t2 = MPI_Wtime();
        currentTrialTime = t2 - t1;
        // #ifdef DEBUG
        // if (printFlag) fprintf(outputFile, "block %d, trial time %.3f\n", blockSize, currentTrialTime);
        // #endif
        
        if (nuChi0EigsRemainAmounts) { // still some Sternheimer eq.s remain
            if (SternBlockSize > 1) {
                if (currentTrialTime > previousTrialTime*2.0) { // previous block size is the best!
                    trySizeFlag = 0;
                    SternBlockSize /= 2;
                } else { // try larger block size
                    SternBlockSize *= 2;
                }
            } else { // at the end of first iteration. Try a larger one!
                SternBlockSize *= 2;
            }
        } else { // all Sternheimer eq.s are done!
            trySizeFlag = 0;
            SternBlockSize = (blockSize > SternBlockSize / 2) ? blockSize : (SternBlockSize / 2); // find the largest block size ever used
        }
        
        sumLhsfunTime += lhsfunTime;
        sumSolveMuTime += solveMuTime;
        sumMultipleTime += multipleTime;
    }
    t6 = MPI_Wtime();

    // use the optimum block size to solve remain Sternheimer eq.s
    int loopTime = nuChi0EigsRemainAmounts / SternBlockSize; // if all Sternheimer eq.s are done in the previous step, loopTime should be zero
    if (nuChi0EigsRemainAmounts % SternBlockSize) loopTime++;
    #ifdef DEBUG
    double t3 = MPI_Wtime();
    #endif
    for (int loop = 0; loop < loopTime; loop++) {
        blockSize = (nuChi0EigsAmounts - startIndex) > SternBlockSize ? SternBlockSize : (nuChi0EigsAmounts - startIndex);
        int maxIter = pSPARC->Nd_d_dmcomm / blockSize;
        double *deltaPsisRealLoop = deltaPsisReal + startIndex*DMnd;
        double *deltaPsisImagLoop = deltaPsisImag + startIndex*DMnd;
        double _Complex *SternheimerRhsLoop = SternheimerRhs + startIndex*DMnd;
        double *resNormRecords = (double *)calloc(sizeof(double), (maxIter + 1) * blockSize); // 1000 is maximum iteration time
        double lhsfunTime = 0.0; double solveMuTime = 0.0; double multipleTime = 0.0;
        int iterTime = block_COCG(lhsfun, pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisRealLoop, deltaPsisImagLoop, SternheimerRhsLoop, blockSize, 
            sternRelativeResTol, maxIter, resNormRecords, &lhsfunTime, &solveMuTime, &multipleTime);
        startIndex += blockSize;
        free(resNormRecords);
        sumLhsfunTime += lhsfunTime;
        sumSolveMuTime += solveMuTime;
        sumMultipleTime += multipleTime;
    }
    #ifdef DEBUG
    double t4 = MPI_Wtime();
    if (printFlag) {
        fprintf(outputFile, "Here are %d blocks for all %d rhs, chosen blockSize %d, spent %.3f ms, sumLhsfunTime %.3f ms, sumSolveMuTime %.3f ms, sumMultipleTime %.3f ms\n", 
            loopTime, nuChi0EigsAmounts, SternBlockSize, ((t6 - t5) + (t4 - t3))*1e3, sumLhsfunTime*1e3, sumSolveMuTime*1e3, sumMultipleTime*1e3);
        fclose(outputFile);
    }
    #endif

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) { // bandWeight includes occupation and spin factor
            deltaRhos[nuChi0EigsIndex * DMnd + i] += 2 * bandWeight * deltaPsisReal[nuChi0EigsIndex * DMnd + i] * (psi[i] / sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    // double _Complex *residual = (double _Complex *)calloc(sizeof(double _Complex), DMnd * nuChi0EigsAmounts);
    // double residualNorm = 0.0;
    // Sternheimer_lhs(pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisReal, deltaPsisImag, residual, nuChi0EigsAmounts);
    // for (int i = 0; i < nuChi0EigsAmounts*DMnd; i++) { // the sum of square of residual norms of all Sternheimer eq.s of the \psi
    //     residual[i] -= SternheimerRhs[i];
    //     residualNorm += conj(residual[i]) * residual[i];
    // }
    free(SternheimerRhs);
    // free(residual);
    // return residualNorm;
}

void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts)
{
    int sg = pSPARC->spin_start_indx + spn_i;
    double dV = pSPARC->dV;
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
    // linear operator Q, only contains the term of psi to be computed, occ2 = occ
    if (flagPQ) {
        for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
            double dotProdPsiXreal = 0.0, dotProdPsiXimag = 0.0;
            for (int i = 0; i < DMnd; i++) {
                dotProdPsiXreal += psi[i]*Xreal[nuChi0EigsIndex*DMnd + i];
                dotProdPsiXimag += psi[i]*Ximag[nuChi0EigsIndex*DMnd + i];
            }
            for (int i = 0; i < DMnd; i++) {
                lhsX[nuChi0EigsIndex*DMnd + i] += psi[i] * (dotProdPsiXreal + dotProdPsiXimag * I) / dV;
            }
        }
    }
    free(lhsXreal_Xcomp);
}

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, int flagCOCGinitial, double *allXorb, double *allLambdas,
                                 double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag, int nuChi0EigsAmounts)
{
    int DMnd = pSPARC->Nd_d_dmcomm;
    int Nstates = pSPARC->Nstates;
    // going to add the code for generating the initial guess based on psis
    if (flagCOCGinitial) {
        double _Complex *deltaPsis = (double _Complex*)calloc(sizeof(double _Complex), DMnd * nuChi0EigsAmounts);
        double _Complex *allXorbComp = (double _Complex*)calloc(sizeof(double _Complex), DMnd * Nstates);
        double _Complex *diagMatrix = (double _Complex*)calloc(sizeof(double _Complex), Nstates*Nstates);
        double _Complex *midVar = (double _Complex*)calloc(sizeof(double _Complex), Nstates*nuChi0EigsAmounts);
        double _Complex *midVar2 = (double _Complex*)calloc(sizeof(double _Complex), Nstates*nuChi0EigsAmounts);
        for (int i = 0; i < Nstates * DMnd; i++) {
            allXorbComp[i] = allXorb[spn_i*DMnd*Nstates + i];
        }
        for (int i = 0; i < Nstates; i++) {
            diagMatrix[i*Nstates + i] = 1.0 / (allLambdas[i] + (double)flagPQ/pSPARC->dV - epsilon - I*omega);
        }
        double _Complex Nalpha = 1.0; double _Complex Nbeta = 0.0;
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nstates, nuChi0EigsAmounts, DMnd,
                  &Nalpha, allXorbComp, DMnd,
                  SternheimerRhs, DMnd, &Nbeta,
                  midVar, Nstates);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nstates, nuChi0EigsAmounts, Nstates,
                  &Nalpha, diagMatrix, Nstates,
                  midVar, Nstates, &Nbeta,
                  midVar2, Nstates);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, Nstates,
                  &Nalpha, allXorbComp, DMnd,
                  midVar2, Nstates, &Nbeta,
                  deltaPsis, DMnd);
        for (int i = 0; i < nuChi0EigsAmounts * DMnd; i++) {
            deltaPsisReal[i] = creal(deltaPsis[i]);
            deltaPsisImag[i] = cimag(deltaPsis[i]);
        }
        free(deltaPsis);
        free(allXorbComp);
        free(diagMatrix);
        free(midVar);
        free(midVar2);
    } else {
        for (int i = 0; i < nuChi0EigsAmounts * DMnd; i++) {
            deltaPsisReal[i] = 0.0;
            deltaPsisImag[i] = 0.0;
        }
    }
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