#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"
#include "blacs.h"     // Cblacs_*
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>

#include "eigenSolver.h"
#include "tools.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "cheFSI.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "tools_RPA.h"

void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double maxEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag) {
    double e = (maxEig - lambdaCutoff) / 2.0;
    double c = (lambdaCutoff + maxEig) / 2.0;
    double sigma = e / (c - minEig);
    double sigmaNew;
    double tau = 2.0 / sigma;
    int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    int totalLength = nuChi0EigsAmount * pSPARC->Nd_d;

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
                char beforeFilterName[100];
                snprintf(beforeFilterName, 100, "nuChi0Eigscomm%d_psis_dVs_beforeFiltering.txt", pRPA->nuChi0EigscommIndex);
                FILE *outputDVs = fopen(beforeFilterName, "w");
                if (outputDVs ==  NULL) {
                    printf("error printing deltaVs_beforeFiltering\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        for (int psiIndex = 0; psiIndex < pSPARC->Nband_bandcomm; psiIndex++) {
                            fprintf(outputDVs, "%12.9f ", pSPARC->Xorb[psiIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputDVs, "\n");
                    }
                    fprintf(outputDVs, "\n");
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            fprintf(outputDVs, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputDVs, "\n");
                    }
                    fprintf(outputDVs, "\n");
                }
                fclose(outputDVs);
            }
        }
    }

    nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, Xs, Ys, nuChi0EigsAmount, flagNoDmcomm);
    for (int index = 0; index < totalLength; index++) {
        Ys[index] = (Ys[index] - c*Xs[index]) * (sigma/e);
    }

    for (int time = 0; time < chebyshevDegree; time++) {
        sigmaNew = 1.0 / (tau - sigma);
        nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, Ys, Yt, nuChi0EigsAmount, flagNoDmcomm);
        for (int index = 0; index < totalLength; index++) {
            Yt[index] = (Yt[index] - c*Ys[index])*(2.0*sigmaNew/e) - (sigma*sigmaNew)*Xs[index];
        }
        memcpy(Xs, Ys, sizeof(double)*totalLength);
        memcpy(Ys, Yt, sizeof(double)*totalLength);
        sigma = sigmaNew;
    }
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            double vec2norm;
            Vector2Norm(Ys + i*pSPARC->Nd_d, pSPARC->Nd_d, &vec2norm, pSPARC->dmcomm_phi);
            VectorScale(Ys + i*pSPARC->Nd_d, pSPARC->Nd_d, 1.0/vec2norm, pSPARC->dmcomm_phi); // unify the length of \Delta V
        }
    }

    if (printFlag) {
        for (int i = 0; i < nuChi0EigsAmount; i++) {
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, pRPA->Ys_phi + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                char afterFilterName[100];
                snprintf(afterFilterName, 100, "nuChi0Eigscomm%d_Ys_afterFiltering.txt", pRPA->nuChi0EigscommIndex);
                FILE *outputYs = fopen(afterFilterName, "w");
                if (outputYs ==  NULL) {
                    printf("error printing deltaVs_afterFiltering\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            fprintf(outputYs, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputYs, "\n");
                    }
                }
                fclose(outputYs);
            }
        }
    }
    free(Yt);
}

void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag) {
    if (dmcomm_phi == MPI_COMM_NULL) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int nproc_dmcomm_phi, rankWorld;
    MPI_Comm_size(dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double alpha = 1.0, beta = 0.0;
    double *Y = pRPA->Ys_phi;
    // allocate memory for block cyclic format of the wavefunction
    double *Y_BLCYC, *Y_BLCYC2;
    
    /* Calculate Mp = Y' * Y */
    t1 = MPI_Wtime();
    Y_BLCYC = pRPA->Ys_phi_BLCYC;
    Y_BLCYC2 = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
    if (size_blacscomm > 1) {
        // distribute Ys into block cyclic format
        pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, Y, &ONE, &ONE, pRPA->desc_orbitals,
                  Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); 
    }
    memcpy(Y_BLCYC2, Y_BLCYC, pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
    t2 = MPI_Wtime();  
#ifdef DEBUG  
    if(!rankWorld) 
        printf("global rank = %2d, Distribute Y to block cyclic format took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3);          
#endif
    t1 = MPI_Wtime();
    // perform matrix multiplication using ScaLAPACK routines
    if (size_blacscomm > 1) { 
#ifdef DEBUG    
        if (!rankWorld) printf("global rank = %d, STARTING PDGEMM ...\n",rankWorld);
#endif   
        // perform matrix multiplication using ScaLAPACK routines
        pdgemm_("T", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
                Y_BLCYC2, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                &beta, pRPA->Mp, &ONE, &ONE, pRPA->desc_Mp_BLCYC); // to get a complete Mp matrix.
                // The generalized eigenproblem Hp X = Mp X \Lambda in RPA has non-symmetric matrix Hp, so matrix Mp should be complete
    } else {
#ifdef DEBUG    
        if (!rankWorld) printf("global rank = %d, STARTING DGEMM...\n",rankWorld);
#endif   
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            1.0, Y_BLCYC2, DMndspe, Y_BLCYC, DMndspe, 
            0.0, pRPA->Mp, pRPA->nuChi0Neig
        ); // to get a complete Mp matrix
        // The generalized eigenproblem Hp X = Mp X \Lambda in RPA has non-symmetric matrix Hp, so matrix Mp should be complete
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) 
        printf("global rank = %2d, YT'*Y in block cyclic format in each blacscomm took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3); 
#endif

    t1 = MPI_Wtime();
    if (nproc_dmcomm_phi > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, pRPA->Mp, pRPA->nr_Mp_BLCYC*pRPA->nc_Mp_BLCYC, 
                      MPI_DOUBLE, MPI_SUM, dmcomm_phi);
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, Allreduce to sum YT'*Y over dmcomm took %.3f ms\n", 
                 rankWorld, (t2 - t1)*1e3); 
#endif
    free(Y_BLCYC2);
    if (printFlag && (pRPA->nr_Hp_BLCYC == pRPA->nuChi0Neig)) {
        int dmcomm_phiRank;
        MPI_Comm_rank(dmcomm_phi, &dmcomm_phiRank);
        if (!dmcomm_phiRank){
            FILE *Mpfile = fopen("Mp.txt", "w");
            for (int row = 0; row < pRPA->nr_Mp_BLCYC; row++) {
                for (int col = 0; col < pRPA->nc_Mp_BLCYC; col++) {
                    fprintf(Mpfile, "%12.9f ", pRPA->Mp[col*pRPA->nr_Mp_BLCYC + row]);
                }
                fprintf(Mpfile, "\n");
            }
            fclose(Mpfile);
        }
    }
#ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, YT*Y used %.3lf ms\n", 1000.0 * (et - st)); 
#endif
}

void Y_orth_gamma(RPA_OBJ* pRPA, int DMnd, int Nspinor_eig) { // If ScaLapack is selected to solve eigenpairs of YT*\nu\Chi0\Y,
// then Y should be orthogonalized, since ScaLapack does not support nonsymmetric generalized eigenvalue problems
// We have to use ScaLapack to solve nonsymmetric eigenvalue problems AX = X\Lambda
    double t1, t2, t3;
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;
    // Orthogonalization using Choleskey 
    t1 = MPI_Wtime();
    Chol_orth(pRPA->Ys_phi_BLCYC, pRPA->desc_orb_BLCYC, pRPA->Mp, pRPA->desc_Mp_BLCYC, &DMndspe, &pRPA->nuChi0Neig);
    t2 = MPI_Wtime();
    // update Ys_phi
    pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, pRPA->Ys_phi_BLCYC, &ONE, &ONE, 
          pRPA->desc_orb_BLCYC, pRPA->Ys_phi, &ONE, &ONE, 
          pRPA->desc_orbitals, &pRPA->ictxt_blacs);
    t3 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank %d, Orthogonalization of orbitals took: %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
    if(!rankWorld) printf("global rank %d, Updating orbitals took: %.3f ms\n", rankWorld, (t3 - t2)*1e3);
#endif
}

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag) {
    if (dmcomm_phi == MPI_COMM_NULL) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int nproc_dmcomm_phi, rankWorld;
    MPI_Comm_size(dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double alpha = 1.0, beta = 0.0;
    double *HY = pRPA->deltaVs_phi;
    // allocate memory for block cyclic format of the wavefunction
    double *Y_BLCYC = pRPA->Ys_phi_BLCYC;
    double *HY_BLCYC;
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // distribute HY
        HY_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        assert(HY_BLCYC != NULL);
        pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, HY, &ONE, &ONE, 
                  pRPA->desc_orbitals, HY_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                  &pRPA->ictxt_blacs);
    } else {
        HY_BLCYC = HY;
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, distributing HY into block cyclic form took %.3f ms\n", 
                 rankWorld, (t2 - t1)*1e3);  
#endif
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // perform matrix multiplication Y' * HY using ScaLAPACK routines
        pdgemm_("T", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
                Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                HY_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                &beta, pRPA->Hp, &ONE, &ONE, pRPA->desc_Hp_BLCYC);
    } else {
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            1.0, Y_BLCYC, DMndspe, HY_BLCYC, DMndspe, 
            0.0, pRPA->Hp, pRPA->nuChi0Neig
        );
    }

    if (nproc_dmcomm_phi > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, pRPA->Hp, pRPA->nr_Hp_BLCYC*pRPA->nc_Hp_BLCYC, 
                      MPI_DOUBLE, MPI_SUM, dmcomm_phi);
    }

    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, finding Y'*HY took %.3f ms\n",rankWorld,(t2-t1)*1e3); 
#endif
    if (size_blacscomm > 1) {
        free(HY_BLCYC);
    }

    if (printFlag && (pRPA->nr_Hp_BLCYC == pRPA->nuChi0Neig)) {
        int dmcomm_phiRank;
        MPI_Comm_rank(dmcomm_phi, &dmcomm_phiRank);
        if (!dmcomm_phiRank){
            FILE *Hpfile = fopen("Hp.txt", "w");
            for (int row = 0; row < pRPA->nr_Hp_BLCYC; row++) {
                for (int col = 0; col < pRPA->nc_Hp_BLCYC; col++) {
                    fprintf(Hpfile, "%12.9f ", pRPA->Hp[col*pRPA->nr_Hp_BLCYC + row]);
                }
                fprintf(Hpfile, "\n");
            }
            fclose(Hpfile);
        }
    }
#ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, project_nuChi0 used %.3lf ms\n", 1000.0 * (et - st)); 
#endif
// // #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}


void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int *signImag, int printFlag) {
    if (dmcomm_phi == MPI_COMM_NULL) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int nproc_dmcomm_phi, rankWorld;
    MPI_Comm_size(dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    double t1, t2;
#ifdef DEBUG    
    double st = MPI_Wtime();
#endif
    if (pRPA->eig_useLAPACK == 1) { // solve generalized eigenproblem Hp X = Mp X \Lambda, where Hp and Mp are non-symmetric
        int info = 0;
        t1 = MPI_Wtime();
        if ((pRPA->nr_Hp_BLCYC == pRPA->nuChi0Neig) && (pRPA->nc_Hp_BLCYC == pRPA->nuChi0Neig)) {
            double *alphar = (double *)malloc(sizeof(double) * pRPA->nuChi0Neig);
            double *alphai = (double *)malloc(sizeof(double) * pRPA->nuChi0Neig);
            double *beta = (double *)malloc(sizeof(double) * pRPA->nuChi0Neig);
            int *sortedIndex = (int *)malloc(sizeof(int) * pRPA->nuChi0Neig);
            double *vl = (double *)malloc(sizeof(double) * pRPA->nr_Hp_BLCYC * pRPA->nc_Hp_BLCYC);
            int ldvl =  pRPA->nuChi0Neig;
            info = LAPACKE_dggev(LAPACK_COL_MAJOR, 'N', 'V', pRPA->nuChi0Neig, pRPA->Hp, pRPA->nuChi0Neig, pRPA->Mp, pRPA->nuChi0Neig, 
                alphar, alphai, beta, vl, ldvl, pRPA->RRnuChi0EigVecs, pRPA->nuChi0Neig);
            for (int i = 0; i < pRPA->nuChi0Neig; i++) {
                alphar[i] = alphar[i] / beta[i];
                if (alphai[i] / beta[i] > 1e-5) *signImag = 1;
            }
            Sort(alphar, pRPA->nuChi0Neig, pRPA->RRnuChi0Eigs, sortedIndex);
            printf("global rank %d, the first 3 eigenvalues are %.5E, %.5E, %.5E\n", rankWorld, pRPA->RRnuChi0Eigs[0], pRPA->RRnuChi0Eigs[1], pRPA->RRnuChi0Eigs[2]);
            int dmcomm_phiRank;
            MPI_Comm_rank(dmcomm_phi, &dmcomm_phiRank);
            if (!dmcomm_phiRank){
                FILE *eigfile = fopen("eig.txt", "a");
                for (int col = 0; col < pRPA->nuChi0Neig; col++) {
                    fprintf(eigfile, "%12.9f ", pRPA->RRnuChi0Eigs[col]);
                }
                fprintf(eigfile, "\n");
                fclose(eigfile);
            }
            if (printFlag) {
                if (!dmcomm_phiRank){
                    FILE *eigVecfile = fopen("eigVec.txt", "w");
                    for (int row = 0; row < pRPA->nr_Hp_BLCYC; row++) {
                        for (int col = 0; col < pRPA->nc_Hp_BLCYC; col++) {
                            fprintf(eigVecfile, "%12.9f ", pRPA->RRnuChi0EigVecs[col*pRPA->nr_Hp_BLCYC + row]);
                        }
                        fprintf(eigVecfile, "\n");
                    }
                    fclose(eigVecfile);
                }
            }
            free(alphar);
            free(alphai);
            free(beta);
            free(sortedIndex);
            free(vl);
        }
        t2 = MPI_Wtime();
#ifdef DEBUG
        if(!rankWorld) {
            printf("==standard eigenproblem: "
                "info = %d, solving standard eigenproblem using LAPACKE_dggev: %.3f ms\n", 
                info, (t2 - t1)*1e3);
        }
#endif
        int ONE = 1;
        t1 = MPI_Wtime();
        // distribute eigenvectors to block cyclic format
        pdgemr2d_(&pRPA->nuChi0Neig, &pRPA->nuChi0Neig, pRPA->RRnuChi0EigVecs, &ONE, &ONE, 
                pRPA->desc_Hp_BLCYC, pRPA->Q, &ONE, &ONE, 
                pRPA->desc_Q_BLCYC, &pRPA->ictxt_blacs_topo);
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rankWorld) {
            printf("==generalized eigenproblem: "
                "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
                (t2 - t1)*1e3);
        }
        #endif
    } else { // solve eigenproblem Hp X = X \Lambda, where Hp is non-symmetric
        // pdgeevx_ // MKL
        // pcgeevx_
    }
    

// #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)

// #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}

void subspace_rotation_unify_eigVecs_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int printFlag) {
    if (dmcomm_phi != MPI_COMM_NULL) {
        // #if defined(USE_MKL) || defined(USE_SCALAPACK)
#ifdef DEBUG
        double st = MPI_Wtime();
#endif
        int rankWorld;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
        int size_blacscomm = pRPA->npnuChi0Neig;

        int ONE = 1;
        int DMndspe = DMnd * Nspinor_eig;

        double alpha = 1.0, beta = 0.0;
        double t1, t2;

        t1 = MPI_Wtime();
        double *Y_BLCYC = pRPA->Ys_phi_BLCYC;
        double *Q = pRPA->Q;
        double *X_BLCYC;
        if (size_blacscomm > 1) {
            // perform matrix multiplication Psi * Q using ScaLAPACK routines
            X_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            pdgemm_("N", "N", &DMndspe, &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &alpha, 
                    Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, Q, &ONE, &ONE, 
                    pRPA->desc_Q_BLCYC, &beta, X_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC);
        } else {
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                DMndspe, pRPA->nuChi0Neig, pRPA->nuChi0Neig, 
                1.0, Y_BLCYC, DMndspe, Q, pRPA->nuChi0Neig,
                0.0, rotatedEigVecs, DMndspe
            );
        }
        t2 = MPI_Wtime();
#ifdef DEBUG
        if(!rankWorld) printf("global rank = %2d, subspace rotation took %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
#endif

        t1 = MPI_Wtime();
        if (size_blacscomm > 1) {
            // distribute rotated orbitals from block cyclic format back into 
            // original format (band + domain)
            pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, X_BLCYC, &ONE, &ONE, 
                      pRPA->desc_orb_BLCYC, rotatedEigVecs, &ONE, &ONE, 
                      pRPA->desc_orbitals, &pRPA->ictxt_blacs);
            free(X_BLCYC);
        }
        t2 = MPI_Wtime();    
#ifdef DEBUG
        if(!rankWorld) 
            printf("global rank = %2d, Distributing orbital back into band + domain format took %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
#endif

#ifdef DEBUG
        double et = MPI_Wtime();
        if (!rankWorld) printf("global rank = %d, Subspace_Rotation used %.3lf ms\n\n", rankWorld, 1000.0 * (et - st));
#endif
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            double vec2norm;
            Vector2Norm(rotatedEigVecs + i*DMndspe, DMndspe, &vec2norm, dmcomm_phi);
            VectorScale(rotatedEigVecs + i*DMndspe, DMndspe, 1.0/vec2norm, dmcomm_phi); // unify the length of \Delta V
        }
// #endif // (USE_MKL or USE_SCALAPACK)
    }

    if (printFlag) {
        int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
        for (int i = 0; i < nuChi0EigsAmount; i++) {
            Transfer_Veff_loc_RPA(pSPARC, pRPA->nuChi0Eigscomm, rotatedEigVecs + i*pSPARC->Nd_d, pRPA->deltaVs + i * pSPARC->Nd_d_dmcomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                char afterFilterName[100];
                snprintf(afterFilterName, 100, "nuChi0Eigscomm%d_deltaVs_afterCheFSI.txt", pRPA->nuChi0EigscommIndex);
                FILE *outputYs = fopen(afterFilterName, "w");
                if (outputYs ==  NULL) {
                    printf("error printing deltaVs_afterCheFSI\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            fprintf(outputYs, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputYs, "\n");
                    }
                }
                fclose(outputYs);
            }
        }
    }
}