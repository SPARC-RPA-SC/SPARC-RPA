#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "cheFSI.h"
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
        }
    }
    else {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                SeededRandVec_complex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->DMVertices, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm_complex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScaleComplex(pRPA->deltaVs_kpt_phi + i*pSPARC->Nd_d, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi); // unify the length of \Delta V
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
        int Nd_d_dmcomm = pSPARC->Nd_d_dmcomm;
        if (pSPARC->isGammaPoint) {
            for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_phi + nuChi0EigIndex*pSPARC->Nd_d, pRPA->deltaVs + nuChi0EigIndex*pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
            }
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
                            for (int index = 0; index < Nd_d_dmcomm; index++) {
                                fprintf(output1stDV, "%12.9f\n", pRPA->deltaVs[nuChi0EigIndex*Nd_d_dmcomm + index]);
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
            if (!flagNoDmcomm) {
                sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, printFlag);
            }
            collect_transfer_deltaRho_gamma(pSPARC, pRPA->deltaRhos, pRPA->deltaRhos_phi, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            Calculate_deltaRhoPotential_gamma(pSPARC, pRPA->deltaRhos_phi, pRPA->deltaVs_phi, nuChi0EigsAmount, printFlag, pRPA->deltaVs, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
        } else {
            for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                Transfer_Veff_loc_RPA_kpt(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_kpt_phi + nuChi0EigIndex*pSPARC->Nd_d, pRPA->deltaVs_kpt + nuChi0EigIndex*pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
            }
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
                            for (int index = 0; index < Nd_d_dmcomm; index++) {
                                fprintf(output1stDV, "%12.9f %12.9f\n", creal(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]), cimag(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]));
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
            if (!flagNoDmcomm) {
                sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, printFlag);
            }
            collect_transfer_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, pRPA->deltaRhos_kpt_phi, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            double qptx = pRPA->q1[qptIndex]; double qpty = pRPA->q2[qptIndex]; double qptz = pRPA->q3[qptIndex];
            Calculate_deltaRhoPotential_kpt(pSPARC, pRPA->deltaRhos_kpt_phi, pRPA->deltaVs_kpt_phi, qptx, qpty, qptz, nuChi0EigsAmount, printFlag, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
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


void cheFSI_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {
    if (!pSPARC->isGammaPoint) {
        printf("RPA kpt is under development\n");
        return;
    }
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
    double lambdaCutoff = -0.01;
    double tolErpaTermConverge = pRPA->tol_ErpaConverge * pRPA->qptWts[qptIndex] * pRPA->omegaWts[omegaIndex];
    int flagCheb = 1;
    int ncheb = 0;
    int chebyshevDegree = 2;
    int printFlag = (nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1);
    // while (flagCheb) {
        chebyshev_filtering(pSPARC, pRPA, qptIndex, omegaIndex, minEig, lambdaCutoff, chebyshevDegree, flagNoDmcomm, printFlag);
    // }
}

double find_min_eigenvalue(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm) { // find min eigenvalue by power method
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double minEig = 0.0;
    double vec2norm = 1.0;
    int loopFlag = 1;
    int maxIter = 30;
    int iter = 0;
    int nuChi0EigsAmount = 1;
    while (loopFlag) {
        if (pSPARC->isGammaPoint) {
            if ((fabs(-vec2norm - minEig) < 2e-4) || (iter == maxIter)) {
                loopFlag = 0;
            }
            minEig = -vec2norm;
            nuChi0_multiply_DeltaV_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs_phi, pRPA->deltaVs_phi, nuChi0EigsAmount, flagNoDmcomm);
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm(pRPA->deltaVs_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScale(pRPA->deltaVs_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
        } else {
            if ((fabs(-vec2norm - minEig) < 2e-4) || (iter == maxIter) || (!pSPARC->isGammaPoint)) { // current kpt calculation is not available
                loopFlag = 0;
            }
            minEig = -vec2norm;
            nuChi0_multiply_DeltaV_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->deltaVs_kpt_phi, pRPA->deltaVs_kpt_phi, nuChi0EigsAmount, flagNoDmcomm);
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm_complex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScaleComplex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
        }
        iter++;
    }
    if (iter == maxIter) printf("qpt %d, omega %d, minimum eigenvalue does not reach the required accuracy.\n", qptIndex, omegaIndex);
    #ifdef DEBUG
    if (!rank) printf("qpt %d, omega %d, iterated for %d times, found minEig %.5E\n", qptIndex, omegaIndex, iter, minEig);
    #endif
    return minEig;
}

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double minEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag) {
    double maxEig = -minEig;
    double e = (maxEig - lambdaCutoff) / 2.0;
    double c = (lambdaCutoff + maxEig) / 2.0;
    double sigma = e / (c - minEig);
    double sigmaNew;
    double tau = 2.0 / sigma;
    int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    int totalLength = nuChi0EigsAmount * pSPARC->Nd_d;

    int innerPrintFlag = 0;
    if (pSPARC->isGammaPoint) {
        double *Xs = pRPA->deltaVs_phi;
        double *Ys = pRPA->Ys_phi;
        double *Yt = (double*)calloc(sizeof(double), totalLength);

        if (printFlag) {
            for (int i = 0; i < nuChi0EigsAmount; i++) {
                Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->deltaVs_phi + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm);
            }
            if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *outputDVs = fopen("deltaVs_beforeFiltering.txt", "w");
                    if (outputDVs ==  NULL) {
                        printf("error printing deltaVs_beforeFiltering\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                                fprintf(outputDVs, "%12.9f\n", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                            }
                            fprintf(outputDVs, "\n");
                        }
                    }
                    fclose(outputDVs);
                }
            }
        }

        nuChi0_multiply_DeltaV_gamma(pSPARC, pRPA, omegaIndex, Xs, Ys, nuChi0EigsAmount, flagNoDmcomm);
        for (int index = 0; index < totalLength; index++) {
            Ys[index] = (Ys[index] - c*Xs[index]) * (sigma/e);
        }

        for (int time = 0; time < chebyshevDegree; time++) {
            sigmaNew = 1.0 / (tau - sigma);
            nuChi0_multiply_DeltaV_gamma(pSPARC, pRPA, omegaIndex, Ys, Yt, nuChi0EigsAmount, flagNoDmcomm);
            for (int index = 0; index < totalLength; index++) {
                Yt[index] = (Yt[index] - c*Ys[index])*(2.0*sigmaNew/e) - (sigma*sigmaNew)*Xs[index];
            }
            memcpy(Xs, Ys, sizeof(double)*totalLength);
            memcpy(Ys, Yt, sizeof(double)*totalLength);
            sigma = sigmaNew;
        }

        if (printFlag) {
            for (int i = 0; i < nuChi0EigsAmount; i++) {
                Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->Ys_phi + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm);
            }
            if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *outputDVs = fopen("deltaVs_afterFiltering.txt", "w");
                    if (outputDVs ==  NULL) {
                        printf("error printing deltaVs_afterFiltering\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                                fprintf(outputDVs, "%12.9f\n", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                            }
                            fprintf(outputDVs, "\n");
                        }
                    }
                    fclose(outputDVs);
                }
            }
        }
        
        free(Yt);
    } else {
        double _Complex *Xs = pRPA->deltaVs_kpt_phi;
        double _Complex *Ys = pRPA->Ys_kpt_phi;
        double _Complex *Yt = (double _Complex*)calloc(sizeof(double _Complex), totalLength);

        nuChi0_multiply_DeltaV_kpt(pSPARC, pRPA, qptIndex, omegaIndex, Xs, Ys, nuChi0EigsAmount, flagNoDmcomm);
        for (int index = 0; index < totalLength; index++) {
            Ys[index] = (Ys[index] - c*Xs[index]) * (sigma/e);
        }

        for (int time = 0; time < chebyshevDegree; time++) {
            sigmaNew = 1.0 / (tau - sigma);
            nuChi0_multiply_DeltaV_kpt(pSPARC, pRPA, qptIndex, omegaIndex, Ys, Yt, nuChi0EigsAmount, flagNoDmcomm);
            for (int index = 0; index < totalLength; index++) {
                Yt[index] = (Yt[index] - c*Ys[index])*(2.0*sigmaNew/e) - (sigma*sigmaNew)*Xs[index];
            }
            memcpy(Xs, Ys, sizeof(double _Complex)*totalLength);
            memcpy(Ys, Yt, sizeof(double _Complex)*totalLength);
            sigma = sigmaNew;
        }
        free(Yt);
    }
}

void nuChi0_multiply_DeltaV_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs_phi, double *nuChi0DVs_phi, int nuChi0EigsAmount, int flagNoDmcomm) {
    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
        Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, DVs_phi + nuChi0EigIndex*pSPARC->Nd_d, pRPA->deltaVs + nuChi0EigIndex*pSPARC->Nd_d_dmcomm); // it tansfer \Delta V at here
    }
    if (!flagNoDmcomm) {
        sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, 0);
    }
    collect_transfer_deltaRho_gamma(pSPARC, pRPA->deltaRhos, pRPA->deltaRhos_phi, nuChi0EigsAmount, 0, pRPA->nuChi0Eigscomm);
    Calculate_deltaRhoPotential_gamma(pSPARC, pRPA->deltaRhos_phi, nuChi0DVs_phi, nuChi0EigsAmount, 0, pRPA->deltaVs, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
}

void nuChi0_multiply_DeltaV_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt_phi, double _Complex *nuChi0DVs_kpt_phi, int nuChi0EigsAmount, int flagNoDmcomm) {
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