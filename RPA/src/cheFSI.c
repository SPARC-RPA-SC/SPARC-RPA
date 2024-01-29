#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "cheFSI.h"
#include "sternheimerEquation.h"
#include "electrostatics_RPA.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "eigenSolverKpt_RPA.h"
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
        int nuChi0EigsAmount = 1;// pRPA->nNuChi0Eigscomm;
        if (qptIndex > pRPA->Nqpts_sym - 1)
            qptIndex = pRPA->Nqpts_sym - 1;
        if (omegaIndex > pRPA->Nomega - 1)
            omegaIndex = pRPA->Nomega - 1;
        int Nd_d_dmcomm = pSPARC->Nd_d_dmcomm;
        if (pSPARC->isGammaPoint) {
            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
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
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
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
            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
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
    int outputEigAmount = (pRPA->nuChi0Neig > 5) ? 5 : pRPA->nuChi0Neig;
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"q-point %d (reduced coords %.3f %.3f %.3f), weight %.3f\n omega %d (value %.3f, 0~1 value %.3f, weight %.3f)\n",
            qptIndex, pRPA->q1[qptIndex]*pSPARC->range_x/(2*M_PI), pRPA->q2[qptIndex]*pSPARC->range_y/(2*M_PI), pRPA->q3[qptIndex]*pSPARC->range_z/(2*M_PI), pRPA->qptWts[qptIndex],
            omegaIndex, pRPA->omega[omegaIndex], pRPA->omega01[omegaIndex], pRPA->omegaWts[omegaIndex]);
        fprintf(output_fp,"ncheb | ErpaTerm (Ha)|       First %d eigenvalues of nu chi0        | Timing (s)\n", outputEigAmount);
        fclose(output_fp);
    }

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    double minEig = 0.0;
    if (!nuChi0EigscommIndex) {
        minEig = find_min_eigenvalue(pSPARC, pRPA, qptIndex, omegaIndex, flagNoDmcomm);
    }
    MPI_Bcast(&minEig, 1, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm);
    double maxEig = -minEig;
    double lambdaCutoff = -0.01;
    double qptOmegaWeight = pRPA->qptWts[qptIndex] * pRPA->omegaWts[omegaIndex];
    double tolErpaTermConverge = pRPA->tol_ErpaConverge * qptOmegaWeight;
    int flagCheb = 1;
    int ncheb = 0;
    int signImag = 0;
    int chebyshevDegree = pRPA->ChebDegreeRPA;
    int printFlag = 0;
    double ErpaTerm = 1000.0, lastErpaTerm = 0.0, t1 = 0.0, t2 = 0.0;
    while (flagCheb) {
        t1 = MPI_Wtime();
        if (ncheb) {
            maxEig = -pRPA->RRnuChi0Eigs[0] / 8.0;
            lambdaCutoff = pRPA->RRnuChi0Eigs[pRPA->nuChi0Neig - 1] + 1e-4;
        }
        if (pSPARC->isGammaPoint) {
            chebyshev_filtering_gamma(pSPARC, pRPA, omegaIndex, minEig, maxEig, lambdaCutoff, chebyshevDegree, flagNoDmcomm, printFlag);
            if (pRPA->npnuChi0Neig > 1) {
                pRPA->Ys_phi_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            } else {
                pRPA->Ys_phi_BLCYC = pRPA->Ys_phi;
            }
            YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig, printFlag);
            if (!pRPA->eig_useLAPACK) { // if we use ScaLapack, not Lapack, to solve eigenpairs of YT*\nu\Chi0*Y, we have to orthogonalize Y
                // because ScaLapack does not support solving AX = BX\Lambda with non-symmetric A or B
                Y_orth_gamma(pRPA, pSPARC->Nd_d, pSPARC->Nspinor_eig);
            }
            nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->Ys_phi, pRPA->deltaVs_phi, pRPA->nNuChi0Eigscomm, flagNoDmcomm); // deltaVs_phi = (\nu\chi0)*Ys_phi for saving memory
            project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig, printFlag);
            generalized_eigenproblem_solver_gamma(pRPA, pSPARC->dmcomm_phi, &signImag, printFlag);
            if (signImag > 0) printf("WARNING: omega %d found eigenvalue has large imag part.\n", omegaIndex);
            subspace_rotation_unify_eigVecs_gamma(pSPARC, pRPA, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig,  pRPA->deltaVs_phi, printFlag);
            if (pRPA->npnuChi0Neig > 1) {
                free(pRPA->Ys_phi_BLCYC);
            }
        } else {
            chebyshev_filtering_kpt(pSPARC, pRPA, qptIndex, omegaIndex, minEig, maxEig, lambdaCutoff, chebyshevDegree, flagNoDmcomm, printFlag);
            if (pRPA->npnuChi0Neig > 1) {
                pRPA->Ys_kpt_phi_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            } else {
                pRPA->Ys_kpt_phi_BLCYC = pRPA->Ys_kpt_phi;
            }
            YT_multiply_Y_kpt(pRPA, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig, printFlag);
            if (!pRPA->eig_useLAPACK) { // if we use ScaLapack, not Lapack, to solve eigenpairs of YT*\nu\Chi0*Y, we have to orthogonalize Y
                // because ScaLapack does not support solving AX = BX\Lambda with non-symmetric A or B
                Y_orth_kpt(pRPA, pSPARC->Nd_d, pSPARC->Nspinor_eig);
            }
            nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->Ys_kpt_phi, pRPA->deltaVs_kpt_phi, pRPA->nNuChi0Eigscomm, flagNoDmcomm); // deltaVs_kpt_phi = (\nu\chi0)*Ys_kpt_phi for saving memory
            project_YT_nuChi0_Y_kpt(pRPA, qptIndex, omegaIndex, flagNoDmcomm, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig, pSPARC->isGammaPoint, printFlag);
            generalized_eigenproblem_solver_kpt(pRPA, pSPARC->dmcomm_phi, &signImag, printFlag);
            if (signImag > 0) printf("WARNING: qpt %d, omega %d found eigenvalue has large imag part.\n", qptIndex, omegaIndex);
            subspace_rotation_unify_eigVecs_kpt(pSPARC, pRPA, pSPARC->dmcomm_phi, pSPARC->Nd_d, pSPARC->Nspinor_eig,  pRPA->deltaVs_phi, printFlag);
            if (pRPA->npnuChi0Neig > 1) {
                free(pRPA->Ys_kpt_phi_BLCYC);
            }
        }
        if (!pRPA->nuChi0EigscommIndex) { // rank 0 in a nuChi0Eigscomm must be in dmcomm_phi of this nuChi0Eigscomm
        // if pRPA->nuChi0EigscommIndex == 0, then all processors in its dmcomm_phi contain all the eigenvalues
            if (!rank) { // to make things simple, just ask rank 0 of the 0th nuChi0Eigscomm to compute convergence
                int signImag = -1;
                ErpaTerm = compute_ErpaTerm(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, pRPA->omega01[omegaIndex], qptOmegaWeight, &signImag);
                flagCheb = fabs(ErpaTerm - lastErpaTerm) > tolErpaTermConverge ? 1 : 0;
                if (ncheb == pRPA->maxitFiltering) flagCheb = 0;
                lastErpaTerm = ErpaTerm;
                printf("qpt %d, omega %d, CheFSI loop %d, ErpaTerm %.6f, flagCheb %d\n", qptIndex, omegaIndex, ncheb, ErpaTerm, flagCheb);
            }
            MPI_Bcast(&flagCheb, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
        }
        MPI_Bcast(&flagCheb, 1, MPI_INT, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the flag to all processors of all nuChi0Eigscomms
        MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the eigenvalues to all processors of all nuChi0Eigscomms
        t2 = MPI_Wtime();
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            FILE *output_fp = fopen(pRPA->filename_out,"a");
            fprintf(output_fp,"%5d   %.5f       ", ncheb, ErpaTerm);
            for (int eigIndex = 0; eigIndex < outputEigAmount; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, " %.5f\n", t2 - t1);
            fclose(output_fp);
        }
        ncheb++;
        // if (ncheb == 4) printFlag = 1;
    }
    pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex] = ErpaTerm;
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
        if ((fabs(-vec2norm - minEig) < 2e-4) || (iter == maxIter)) {
            loopFlag = 0;
        }
        minEig = -vec2norm;
        if (pSPARC->isGammaPoint) {
            nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs_phi, pRPA->deltaVs_phi, nuChi0EigsAmount, flagNoDmcomm);
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm(pRPA->deltaVs_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScale(pRPA->deltaVs_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
        } else {
            if (!pSPARC->isGammaPoint) { // current kpt calculation is not available
                loopFlag = 0;
            }
            nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->deltaVs_kpt_phi, pRPA->deltaVs_kpt_phi, nuChi0EigsAmount, flagNoDmcomm);
            if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
                Vector2Norm_complex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
                VectorScaleComplex(pRPA->deltaVs_kpt_phi, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi);
            }
        }
        MPI_Bcast(&vec2norm, 1, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
        iter++;
    }
    if (iter == maxIter) printf("qpt %d, omega %d, minimum eigenvalue does not reach the required accuracy.\n", qptIndex, omegaIndex);
    #ifdef DEBUG
    if (!rank) printf("qpt %d, omega %d, iterated for %d times, found minEig %.5E\n", qptIndex, omegaIndex, iter, minEig);
    #endif
    return minEig;
}

double compute_ErpaTerm(double *RRnuChi0Eigs, int nuChi0Neig, double omegaMesh01, double qptOmegaWeight, int *signImag) {
    double ErpaTerm = 0.0;
    for (int i = 0; i < nuChi0Neig; i++) {
        ErpaTerm += log(1.0 - RRnuChi0Eigs[i]) + RRnuChi0Eigs[i];
    }
    ErpaTerm *= qptOmegaWeight / (omegaMesh01*omegaMesh01) / (2.0*M_PI);
    return ErpaTerm;
}