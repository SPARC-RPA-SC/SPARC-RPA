#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "test_Hx_nuChi0_eigSolver.h"
#include "sternheimerEquation.h"
#include "electrostatics_RPA.h"
#include "nuChi0VecRoutines.h"
#include "tools_RPA.h"
#include "eigenSolverGamma_RPA.h"
#include "eigenSolverKpt_RPA.h"

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

    // if (nuChi0EigscommIndex == 0) { // this test is done in only one nuChi0Eigscomm pRPA->npnuChi0Neig - 1
        int printFlag = 1;
        int qptIndex = 1;
        int omegaIndex = pRPA->Nomega - 1; // 0;
        int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;// 1;
        if (qptIndex > pRPA->Nqpts_sym - 1)
            qptIndex = pRPA->Nqpts_sym - 1;
        if (omegaIndex > pRPA->Nomega - 1)
            omegaIndex = pRPA->Nomega - 1;
        if (pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
            if (pSPARC->isGammaPoint) {
                collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
            } else {
                collect_allXorb_allLambdas_kpt(pSPARC, pRPA, qptIndex);
            }
        }
        int Nd_d_dmcomm = pSPARC->Nd_d_dmcomm;
        if (pSPARC->isGammaPoint) {
            double t1 = MPI_Wtime();
            // if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print all \Delta V vectors of the last nuChi0Eigscomm.
            //     // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
            //     // only one processor print
            //     int dmcommRank;
            //     MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            //     if (dmcommRank == 0) {
            //         FILE *output1stDV = fopen("deltaVs.txt", "w");
            //         if (output1stDV ==  NULL) {
            //             printf("error printing delta Vs in test\n");
            //             exit(EXIT_FAILURE);
            //         } else {
            //             for (int index = 0; index < Nd_d_dmcomm; index++) {
            //                 for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
            //                     fprintf(output1stDV, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*Nd_d_dmcomm + index]);
            //                 }
            //                 fprintf(output1stDV, "\n");
            //             }
            //         }
            //         fclose(output1stDV);
            //     }
            // }
            double *sqrtNuDVs = pRPA->sprtNuDeltaVs;
            Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaVs, sqrtNuDVs, nuChi0EigsAmount, printFlag, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t3 = MPI_Wtime();
            if (!flagNoDmcomm) {
                sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, sqrtNuDVs, printFlag);
            }
            double t4 = MPI_Wtime();
            collect_deltaRho_gamma(pSPARC, pRPA->deltaRhos, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaRhos, pRPA->deltaVs, nuChi0EigsAmount, printFlag, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t2 = MPI_Wtime();
            if (!rank) {
                FILE *outputFile = fopen(pRPA->timeRecordname, "a");
                fprintf(outputFile, "omega %d, sternheimer time %.3f, total Multiplication time is %.3f\n", omegaIndex, (t4 - t3)*1e3, (t2 - t1)*1e3);
                fclose(outputFile);
            }
        } else {
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
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
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
                sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, pRPA->deltaVs_kpt, printFlag);
            }
            collect_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            double qptx = pRPA->q1[qptIndex]; double qpty = pRPA->q2[qptIndex]; double qptz = pRPA->q3[qptIndex];
            Calculate_deltaRhoPotential_kpt(pSPARC, pRPA->deltaRhos_kpt, pRPA->deltaVs_kpt, qptx, qpty, qptz, nuChi0EigsAmount, printFlag, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
        }
    // }
}

void test_Hx(SPARC_OBJ *pSPARC, double *testHxAccuracy)
{
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

void test_eigSolver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    int rank, size;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    MPI_Comm_size(pRPA->nuChi0Eigscomm, &size);
    int printFlag = 1;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    int lengthY = pSPARC->Nd_d_dmcomm*pRPA->nNuChi0Eigscomm;
    if (rank == size - 1) {
        if (pSPARC->isGammaPoint) {
            memcpy(pRPA->Ys, pRPA->deltaVs, sizeof(double)*lengthY); // to replace chebyshev filtering
            if (pRPA->npnuChi0Neig > 1) {
                pRPA->Ys_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            } else {
                pRPA->Ys_BLCYC = pRPA->Ys;
            }
            YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            if (!pRPA->eig_useLAPACK) { // if we use ScaLapack, not Lapack, to solve eigenpairs of YT*\nu\Chi0*Y, we have to orthogonalize Y
                // because ScaLapack does not support solving AX = BX\Lambda with non-symmetric A or B
                Y_orth_gamma(pSPARC, pRPA, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            }
            for (int i = 0; i < lengthY; i++) {
                pRPA->deltaVs[i] = -pRPA->Ys[lengthY - i - 1]; // to replace nuChi0_mult_vectors_gamma // deltaVs = (\nu\chi0)*Ys for saving memory
            }
            printf("rank %d of nuChi0Eigscomm %d, reached line 183 of test_eigSolver\n", rank, pRPA->nuChi0EigscommIndex);
            project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys_BLCYC, pRPA->deltaVs, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            generalized_eigenproblem_solver_gamma(pRPA, pSPARC->dmcomm, pRPA->nuChi0BlacsComm, 10, flagNoDmcomm, printFlag); // pSPARC->eig_paral_blksz
            subspace_rotation_unify_eigVecs_gamma(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, pRPA->deltaVs, flagNoDmcomm, printFlag);
            MPI_Bcast(pRPA->RRoriginSeqEigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the eigenvalues to all processors of all nuChi0Eigscomms
            if (pRPA->npnuChi0Neig > 1) {
                free(pRPA->Ys_BLCYC);
            }
        } else {
            
        }
    }
}