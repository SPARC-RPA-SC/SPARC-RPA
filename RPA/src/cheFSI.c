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
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (flagNoDmcomm) return;
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    int seed_offset = 0;
    double vec2norm = 0.0;
    if (pSPARC->isGammaPoint) {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                SeededRandVec(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                VectorScale(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
        }
    }
    else {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                SeededRandVec_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                VectorScaleComplex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
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
    int outputFirstEigAmount = (pRPA->nuChi0Neig > 2) ? 2 : pRPA->nuChi0Neig;
    int outputLastEigAmount = (pRPA->nuChi0Neig > 2) ? 2 : pRPA->nuChi0Neig;
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"q-point %d (reduced coords %.3f %.3f %.3f), weight %.3f\n omega %d (value %.3f, 0~1 value %.3f, weight %.3f)\n",
            qptIndex + 1, pRPA->q1[qptIndex]*pSPARC->range_x/(2*M_PI), pRPA->q2[qptIndex]*pSPARC->range_y/(2*M_PI), pRPA->q3[qptIndex]*pSPARC->range_z/(2*M_PI), pRPA->qptWts[qptIndex],
            omegaIndex + 1, pRPA->omega[omegaIndex], pRPA->omega01[omegaIndex], pRPA->omegaWts[omegaIndex]);
        fprintf(output_fp,"ncheb|ErpaTerm(Ha/atom)|First %d eigs & Last %d eigs of nu chi0|eig Error|Timing (s)\n", outputFirstEigAmount, outputLastEigAmount);
        fclose(output_fp);
    }

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
        if (pSPARC->isGammaPoint) {
            collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
        } else {
            collect_allXorb_allLambdas_kpt(pSPARC, pRPA, qptIndex);
        }
    }

    double tolRelEigError = pRPA->tol_EigConverge[omegaIndex];
    double iniErrorTime = 0.0, t1 = 0.0, t2 = 0.0;
    double ErpaTerm = 1000.0;
    double qptOmegaWeight = pRPA->qptWts[qptIndex] * pRPA->omegaWts[omegaIndex];
    t1 = MPI_Wtime();
    double initialError = estimate_initialError(pSPARC, pRPA, qptIndex, omegaIndex, flagNoDmcomm); // compute a initial Erpa by projecting the operator on eigenvectors of the previous omega operator
    printf("qpt %d, omega %d, initial relative eig error %.3E, tolRelEigError %.3E\n", qptIndex, omegaIndex, initialError, tolRelEigError);
    t2 = MPI_Wtime();
    iniErrorTime = t2 - t1;
    #ifdef DEBUG
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, estimate_initialError spent %.2E ms.\n", omegaIndex, iniErrorTime);
    }
    #endif
    if (initialError < tolRelEigError) { // if the initial eigenvectors are very good, then no need to do Chebyshev filtering!
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            ErpaTerm = compute_ErpaTerm(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, pRPA->omega01[omegaIndex], qptOmegaWeight);
            FILE *output_fp = fopen(pRPA->filename_out,"a");
            fprintf(output_fp,"% 3d     %.3E    ", 0, ErpaTerm / (double)pSPARC->n_atom);
            for (int eigIndex = 0; eigIndex < outputFirstEigAmount; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, "; ");
            for (int eigIndex = pRPA->nuChi0Neig - outputLastEigAmount; eigIndex < pRPA->nuChi0Neig; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, " %.3E %.2f\n", initialError, iniErrorTime);
            fclose(output_fp);
        }
        pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex] = ErpaTerm;
        return;
    }

    t1 = MPI_Wtime();
    double minEig = 0.0;
    if (!nuChi0EigscommIndex) {
        minEig = find_min_eigenvalue(pSPARC, pRPA, qptIndex, omegaIndex, flagNoDmcomm);
    }
    MPI_Bcast(&minEig, 1, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm);
    double maxEig = -minEig;
    double lambdaCutoff = 0.0;
    t2 = MPI_Wtime();
    double minEigValueTime = t2 - t1;
    #ifdef DEBUG
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, find_min_eigenvalue spent %.2E ms.\n", omegaIndex + 1, minEigValueTime);
    }
    #endif
    
    // double tolErpaTermConverge = pRPA->tol_ErpaConverge * qptOmegaWeight;
    int flagCheb = 1;
    int ncheb = 0;
    int signImag = 0;
    int chebyshevDegree = pRPA->ChebDegreeRPA;
    int printFlag = 0; // for midvariables output
    double error = 0.0;
    double t3, t4;
    double sumFilteringTime = 0.0, sumYT_mul_YTime = 0.0, sumYT_operator_YTime = 0.0, sumEigTime = 0.0, sumRotationTime = 0.0, sumEvaluateErrTime = 0.0;
    while (flagCheb) {
        t1 = MPI_Wtime();
        if (ncheb) {
            minEig = pRPA->RRnuChi0Eigs[0]; // to prevent minEig from power method not converging
            maxEig = -pRPA->RRnuChi0Eigs[0];
            lambdaCutoff = pRPA->RRnuChi0Eigs[pRPA->nuChi0Neig - 1] + 1e-4;
        }
        if (pSPARC->isGammaPoint) {
            #ifdef DEBUG
            t3 = MPI_Wtime();
            #endif
            chebyshev_filtering_gamma(pSPARC, pRPA, omegaIndex, minEig, maxEig, lambdaCutoff, chebyshevDegree, flagNoDmcomm, ncheb, printFlag);
            #ifdef DEBUG
            t4 = MPI_Wtime();
            sumFilteringTime += t4 - t3;
            if ((!rank) && (!nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, chebyshev_filtering spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
            }
            #endif
            if (pRPA->npnuChi0Neig > 1) {
                pRPA->Ys_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            } else {
                pRPA->Ys_BLCYC = pRPA->Ys;
            }
            #ifdef DEBUG
            t3 = MPI_Wtime();
            #endif
            YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            #ifdef DEBUG
            t4 = MPI_Wtime();
            sumYT_mul_YTime += t4 - t3;
            if ((!rank) && (!nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
            }
            t3 = MPI_Wtime();
            #endif
            // if (!pRPA->eig_useLAPACK) { // if we use ScaLapack, not Lapack, to solve eigenpairs of YT*\nu\Chi0*Y, we have to orthogonalize Y
            //     // because ScaLapack does not support solving AX = BX\Lambda with non-symmetric A or B
            //     Y_orth_gamma(pSPARC, pRPA, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            // }
            nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->Ys, pRPA->deltaVs, pRPA->nNuChi0Eigscomm, flagNoDmcomm, printFlag); // deltaVs = (\nu\chi0)*Ys for saving memory
            #ifdef DEBUG
            t4 = MPI_Wtime();
            sumYT_operator_YTime += t4 - t3;
            if ((!rank) && (!nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, nu^0.5 chi0 nu^0.5 multiplying Ys spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
            }
            #endif
            if (!pRPA->nuChi0EigsBridgeCommIndex) {
                #ifdef DEBUG
                t3 = MPI_Wtime();
                #endif
                project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys_BLCYC, pRPA->deltaVs, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
                #ifdef DEBUG
                t4 = MPI_Wtime();
                sumYT_operator_YTime += t4 - t3;
                if ((!rank) && (!nuChi0EigscommIndex)) {
                    printf("omega %d, subspace iteration %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
                }
                t3 = MPI_Wtime();
                #endif
                generalized_eigenproblem_solver_gamma(pRPA, pSPARC->dmcomm, pRPA->nuChi0BlacsComm, pSPARC->eig_paral_blksz, flagNoDmcomm, printFlag); // pSPARC->eig_paral_blksz
                #ifdef DEBUG
                t4 = MPI_Wtime();
                sumEigTime += t4 - t3;
                if ((!rank) && (!nuChi0EigscommIndex)) {
                    printf("omega %d, subspace iteration %d, generalized_eigenproblem_solver spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
                }
                t3 = MPI_Wtime();
                #endif
                subspace_rotation_unify_eigVecs_gamma(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, pRPA->deltaVs, flagNoDmcomm, printFlag);
                #ifdef DEBUG
                t4 = MPI_Wtime();
                sumRotationTime += t4 - t3;
                if ((!rank) && (!nuChi0EigscommIndex)) {
                    printf("omega %d, subspace iteration %d, subspace_rotation spent %.2E ms.\n", omegaIndex + 1, ncheb + 1, (t4 - t3)*1e3);
                }
                #endif
                MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
            }
            MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
            MPI_Bcast(pRPA->deltaVs, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
            #ifdef DEBUG
            t3 = MPI_Wtime();
            #endif
            error = evaluate_cheFSI_error_gamma(pSPARC, pRPA, omegaIndex, flagNoDmcomm);
            #ifdef DEBUG
            t4 = MPI_Wtime();
            sumEvaluateErrTime += t4 - t3;
            #endif
            if (pRPA->npnuChi0Neig > 1) {
                free(pRPA->Ys_BLCYC);
            }
        } else {
            chebyshev_filtering_kpt(pSPARC, pRPA, qptIndex, omegaIndex, minEig, maxEig, lambdaCutoff, chebyshevDegree, flagNoDmcomm, printFlag);
            if (pRPA->npnuChi0Neig > 1) {
                pRPA->Ys_kpt_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
            } else {
                pRPA->Ys_kpt_BLCYC = pRPA->Ys_kpt;
            }
            YT_multiply_Y_kpt(pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            if (!pRPA->eig_useLAPACK) { // if we use ScaLapack, not Lapack, to solve eigenpairs of YT*\nu\Chi0*Y, we have to orthogonalize Y
                // because ScaLapack does not support solving AX = BX\Lambda with non-symmetric A or B
                Y_orth_kpt(pSPARC, pRPA, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, printFlag);
            }
            nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->Ys_kpt, pRPA->deltaVs_kpt, pRPA->nNuChi0Eigscomm, flagNoDmcomm); // deltaVs_kpt = (\nu\chi0)*Ys_kpt for saving memory
            project_YT_nuChi0_Y_kpt(pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, printFlag);
            generalized_eigenproblem_solver_kpt(pRPA, pSPARC->dmcomm, pRPA->nuChi0BlacsComm, pSPARC->eig_paral_blksz, &signImag, printFlag);
            if (signImag > 0) printf("WARNING: qpt %d, omega %d found eigenvalue has large imag part.\n", qptIndex, omegaIndex);
            subspace_rotation_unify_eigVecs_kpt(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig,  pRPA->deltaVs, printFlag);
            MPI_Bcast(pRPA->RRoriginSeqEigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the eigenvalues to all processors of all nuChi0Eigscomms
            error = evaluate_cheFSI_error_kpt(pSPARC, pRPA, qptIndex, omegaIndex, flagNoDmcomm);
            if (pRPA->npnuChi0Neig > 1) {
                free(pRPA->Ys_kpt_BLCYC);
            }
        }
        if (!pRPA->nuChi0EigscommIndex) { // rank 0 in a nuChi0Eigscomm must be in dmcomm_phi of this nuChi0Eigscomm
        // if pRPA->nuChi0EigscommIndex == 0, then all processors in its dmcomm_phi contain all the eigenvalues
            if (!rank) { // to make things simple, just ask rank 0 of the 0th nuChi0Eigscomm to compute convergence
                ErpaTerm = compute_ErpaTerm(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, pRPA->omega01[omegaIndex], qptOmegaWeight);
                flagCheb = error > tolRelEigError ? 1 : 0;
                if (ncheb == pRPA->maxitFiltering - 1) flagCheb = 0;
                printf("qpt %d, omega %d, CheFSI loop %d, ErpaTerm %.6f, relative eig error %.3E, flagCheb %d\n", qptIndex, omegaIndex, ncheb, ErpaTerm, error, flagCheb);
            }
            MPI_Bcast(&flagCheb, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
        }
        MPI_Bcast(&flagCheb, 1, MPI_INT, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the flag to all processors of all nuChi0Eigscomms
        MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the eigenvalues to all processors of all nuChi0Eigscomms
        t2 = MPI_Wtime();
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            FILE *output_fp = fopen(pRPA->filename_out,"a");
            fprintf(output_fp,"% 3d     %.3E    ", ncheb + 1, ErpaTerm / (double)pSPARC->n_atom);
            for (int eigIndex = 0; eigIndex < outputFirstEigAmount; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, "; ");
            for (int eigIndex = pRPA->nuChi0Neig - outputLastEigAmount; eigIndex < pRPA->nuChi0Neig; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, " %.3E %.2f\n", error, iniErrorTime + minEigValueTime + t2 - t1);
            fclose(output_fp);
        }
        ncheb++;
        if (ncheb == 1) {
            printFlag = 0;
            iniErrorTime = 0.0;
            minEigValueTime = 0.0;
        }
        MPI_Barrier(pRPA->nuChi0Eigscomm);
    }
    #ifdef DEBUG
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        printf("omega %d, sumFilteringTime %.3f ms, sumYT_mul_YTime %.3f ms, sumYT_operator_YTime %.3f ms, sumEigTime %.3f ms, sumRotationTime %.3f ms, sumEvaluateErrTime %.3f ms\n",
         omegaIndex, sumFilteringTime*1e3, sumYT_mul_YTime*1e3, sumYT_operator_YTime*1e3, sumEigTime*1e3, sumRotationTime*1e3, sumEvaluateErrTime*1e3);
    }
    #endif
    pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex] = ErpaTerm;
}

double find_min_eigenvalue(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm) { // find min eigenvalue by power method
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double minEig = 0.0;
    double vec2norm = 1.0;
    int loopFlag = 1;
    int maxIter = 100;
    int iter = 0;
    int nuChi0EigsAmount = 1;
    while (loopFlag) {
        if ((fabs(-vec2norm - minEig) < 2e-4) || (iter == maxIter)) {
            loopFlag = 0;
        }
        minEig = -vec2norm;
        if (pSPARC->isGammaPoint) {
            nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs, pRPA->deltaVs, nuChi0EigsAmount, flagNoDmcomm, 0);
            if (!flagNoDmcomm) {
                Vector2Norm(pRPA->deltaVs, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                VectorScale(pRPA->deltaVs, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm);
            }
        } else {
            if (!pSPARC->isGammaPoint) { // current kpt calculation is not available
                loopFlag = 0;
            }
            nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->deltaVs_kpt, pRPA->deltaVs_kpt, nuChi0EigsAmount, flagNoDmcomm);
            if (!flagNoDmcomm) {
                Vector2Norm_complex(pRPA->deltaVs_kpt, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm);
                VectorScaleComplex(pRPA->deltaVs_kpt, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm);
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

double estimate_initialError(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm) {
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double error = 0.0;
    double t1, t2;
    if (pSPARC->isGammaPoint) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs, pRPA->Ys, pRPA->nNuChi0Eigscomm, flagNoDmcomm, 0); // Ys = (\nu\chi0)*deltaVs for saving memory
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
            printf("In function estimate_initialError, omega %d, operator nu^0.5 chi0 nu^0.5 multiplying vectors spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
        }
        #endif
        if (pRPA->npnuChi0Neig > 1) {
            pRPA->Ys_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        } else {
            pRPA->Ys_BLCYC = pRPA->deltaVs;
        }
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->deltaVs, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, 0);
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
            printf("In function estimate_initialError, omega %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
        }
        #endif
        if (!pRPA->nuChi0EigsBridgeCommIndex) {
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif
            project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys_BLCYC, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, 0);
            #ifdef DEBUG
            t2 = MPI_Wtime();
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("In function estimate_initialError, omega %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
            }
            t1 = MPI_Wtime();
            #endif
            generalized_eigenproblem_solver_gamma(pRPA, pSPARC->dmcomm, pRPA->nuChi0BlacsComm, pSPARC->eig_paral_blksz, flagNoDmcomm, 0); // pSPARC->eig_paral_blksz
            #ifdef DEBUG
            t2 = MPI_Wtime();
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("In function estimate_initialError, omega %d, eigenproblem_solver spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
            }
            t1 = MPI_Wtime();
            #endif
            subspace_rotation_unify_eigVecs_gamma(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, pRPA->deltaVs, flagNoDmcomm, 0);
            #ifdef DEBUG
            t2 = MPI_Wtime();
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("In function estimate_initialError, omega %d, subspace_rotation spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
            }
            #endif
            MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
        }
        MPI_Bcast(pRPA->deltaVs, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
        MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        error = evaluate_cheFSI_error_gamma(pSPARC, pRPA, omegaIndex, flagNoDmcomm);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
            printf("In function estimate_initialError, omega %d, evaluate_cheFSI_error spent %.2E ms.\n", omegaIndex + 1, (t2 - t1)*1e3);
        }
        #endif
        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
        if (pRPA->npnuChi0Neig > 1) {
            free(pRPA->Ys_BLCYC);
        }
    } else {

    }
    return error;
}

double compute_ErpaTerm(double *RRnuChi0Eigs, int nuChi0Neig, double omegaMesh01, double qptOmegaWeight) {
    double ErpaTerm = 0.0;
    for (int i = 0; i < nuChi0Neig; i++) {
        ErpaTerm += log(1.0 - RRnuChi0Eigs[i]) + RRnuChi0Eigs[i];
    }
    ErpaTerm *= qptOmegaWeight / (omegaMesh01*omegaMesh01) / (2.0*M_PI);
    return ErpaTerm;
}
