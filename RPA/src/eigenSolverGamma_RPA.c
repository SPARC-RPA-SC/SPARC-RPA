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
#include "linearAlgebra.h"

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "cheFSI.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "tools_RPA.h"

#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))

void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double maxEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int ncheb, int printFlag) {
    if (flagNoDmcomm) return;
    
    double e = (maxEig - lambdaCutoff) / 2.0;
    double c = (lambdaCutoff + maxEig) / 2.0;
    double sigma = e / (c - minEig);
    double sigmaNew;
    double tau = 2.0 / sigma;
    int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    int totalLength = nuChi0EigsAmount * pSPARC->Nd_d_dmcomm;

    double *Xs = pRPA->deltaVs;
    double *Ys = pRPA->Ys;
    double *Yt = (double*)calloc(sizeof(double), totalLength);

    // if (printFlag) {
    //     if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
    //         int dmcommRank;
    //         MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
    //         if (dmcommRank == 0) {
    //             char beforeFilterName[100];
    //             snprintf(beforeFilterName, 100, "nuChi0Eigscomm%d_psis_dVs_beforeFiltering.txt", pRPA->nuChi0EigscommIndex);
    //             FILE *outputDVs = fopen(beforeFilterName, "w");
    //             if (outputDVs ==  NULL) {
    //                 printf("error printing deltaVs_beforeFiltering\n");
    //                 exit(EXIT_FAILURE);
    //             } else {
    //                 fprintf(outputDVs, "maxEig %12.9f, lambdaCutoff %12.9f, minEig %12.9f\n", maxEig, lambdaCutoff, minEig);
    //                 for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                     for (int psiIndex = 0; psiIndex < pSPARC->Nband_bandcomm; psiIndex++) {
    //                         fprintf(outputDVs, "%12.9f ", pSPARC->Xorb[psiIndex*pSPARC->Nd_d_dmcomm + index]);
    //                     }
    //                     fprintf(outputDVs, "\n");
    //                 }
    //                 fprintf(outputDVs, "\n");
    //                 for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                     for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
    //                         fprintf(outputDVs, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
    //                     }
    //                     fprintf(outputDVs, "\n");
    //                 }
    //                 fprintf(outputDVs, "\n");
    //             }
    //             fclose(outputDVs);
    //         }
    //     }
    // }

    if (!ncheb) { // the product has been saved in Ys (from the error evaluation of the last CheFSI)
        nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, Xs, Ys, nuChi0EigsAmount, flagNoDmcomm, printFlag);
    }
    for (int index = 0; index < totalLength; index++) {
        Ys[index] = (Ys[index] - c*Xs[index]) * (sigma/e);
    }

    for (int time = 0; time < chebyshevDegree; time++) {
        sigmaNew = 1.0 / (tau - sigma);
        nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, Ys, Yt, nuChi0EigsAmount, flagNoDmcomm, printFlag);
        for (int index = 0; index < totalLength; index++) {
            Yt[index] = (Yt[index] - c*Ys[index])*(2.0*sigmaNew/e) - (sigma*sigmaNew)*Xs[index];
        }
        memcpy(Xs, Ys, sizeof(double)*totalLength);
        memcpy(Ys, Yt, sizeof(double)*totalLength);
        sigma = sigmaNew;
    }

    for (int i = 0; i < nuChi0EigsAmount; i++) {
        double vec2norm;
        Vector2Norm(Ys + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
        VectorScale(Ys + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
    }

    // if (printFlag) {
    //     if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
    //         int dmcommRank;
    //         MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
    //         if (dmcommRank == 0) {
    //             char afterFilterName[100];
    //             snprintf(afterFilterName, 100, "nuChi0Eigscomm%d_Ys_afterFiltering.txt", pRPA->nuChi0EigscommIndex);
    //             FILE *outputYs = fopen(afterFilterName, "w");
    //             if (outputYs ==  NULL) {
    //                 printf("error printing Ys_afterFiltering\n");
    //                 exit(EXIT_FAILURE);
    //             } else {
    //                 for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                     for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
    //                         fprintf(outputYs, "%12.9f ", Ys[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
    //                     }
    //                     fprintf(outputYs, "\n");
    //                 }
    //             }
    //             fclose(outputYs);
    //         }
    //     }
    // }
    free(Yt);
}

void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm, double *Y, int DMnd, int Nspinor_eig, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi,
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0BlacsComm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double alpha = 1.0, beta = 0.0;
    // double *Y = pRPA->Ys;
    // allocate memory for block cyclic format of the wavefunction
    double *Y_BLCYC, *Y_BLCYC2;
    
    /* Calculate Mp = Y' * Y */
    t1 = MPI_Wtime();
    Y_BLCYC = pRPA->Ys_BLCYC;
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

    free(Y_BLCYC2);
    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Mpname[50];
        snprintf(Mpname, 50, "rank%dblacsComm%d_Mp.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Mpfile = fopen(Mpname, "a");
        for (int row = 0; row < pRPA->nr_Mp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Mp_BLCYC; col++) {
                fprintf(Mpfile, "%12.9f ", pRPA->Mp[col*pRPA->nr_Mp_BLCYC + row]);
            }
            fprintf(Mpfile, "\n");
        }
        fprintf(Mpfile, "\n");
        fclose(Mpfile);
    }
#ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, YT*Y used %.3lf ms\n", 1000.0 * (et - st)); 
#endif
}

void Y_orth_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int DMnd, int Nspinor_eig, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
    double t1, t2, t3;
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;
    // Orthogonalization using Choleskey 
    t1 = MPI_Wtime();
    Chol_orth(pRPA->Ys_BLCYC, pRPA->desc_orb_BLCYC, pRPA->Mp, pRPA->desc_Mp_BLCYC, &DMndspe, &pRPA->nuChi0Neig);
    t2 = MPI_Wtime();
    // update Ys
    pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, pRPA->Ys_BLCYC, &ONE, &ONE, 
          pRPA->desc_orb_BLCYC, pRPA->Ys, &ONE, &ONE, 
          pRPA->desc_orbitals, &pRPA->ictxt_blacs);
    t3 = MPI_Wtime();
    if (printFlag) {
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                char afterOrthoName[100];
                snprintf(afterOrthoName, 100, "nuChi0Eigscomm%d_Ys_afterOrtho.txt", pRPA->nuChi0EigscommIndex);
                FILE *outputYs = fopen(afterOrthoName, "w");
                if (outputYs ==  NULL) {
                    printf("error printing deltaVs_afterOrtho\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int index = 0; index < DMnd; index++) {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            fprintf(outputYs, "%12.9f ", pRPA->Ys[nuChi0EigIndex*DMnd + index]);
                        }
                        fprintf(outputYs, "\n");
                    }
                }
                fclose(outputYs);
            }
        }
    }
#ifdef DEBUG
    if(!rankWorld) printf("global rank %d, Orthogonalization of orbitals took: %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
    if(!rankWorld) printf("global rank %d, Updating orbitals took: %.3f ms\n", rankWorld, (t3 - t2)*1e3);
#endif
}

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm, double *Y_BLCYC, double *HY, int DMnd, int Nspinor_eig, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi, 
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0BlacsComm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double alpha = 1.0, beta = 0.0;
    // double *HY = pRPA->deltaVs;
    // allocate memory for block cyclic format of the wavefunction
    // double *Y_BLCYC = pRPA->Ys_BLCYC;
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

    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, finding Y'*HY took %.3f ms\n",rankWorld,(t2-t1)*1e3); 
#endif
    if (size_blacscomm > 1) {
        free(HY_BLCYC);
    }

    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Hpname[50];
        snprintf(Hpname, 50, "rank%dblacsComm%d_Hp.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Hpfile = fopen(Hpname, "a");
        for (int row = 0; row < pRPA->nr_Hp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Hp_BLCYC; col++) {
                fprintf(Hpfile, "%12.9f, ", pRPA->Hp[col*pRPA->nr_Hp_BLCYC + row]);
            }
            fprintf(Hpfile, "\n");
        }
        fprintf(Hpfile, "\n");
        fclose(Hpfile);
    }
#ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, project_nuChi0 used %.3lf ms\n", 1000.0 * (et - st)); 
#endif
// // #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}


void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm, MPI_Comm blacsComm, int blksz, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi, 
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    double t1, t2;
#ifdef DEBUG    
    double st = MPI_Wtime();
#endif
    if (pRPA->eig_useLAPACK == 1) { // solve generalized eigenproblem Hp X = Mp X \Lambda, where Hp and Mp are non-symmetric
        int info = 0;
        t1 = MPI_Wtime();
        if ((pRPA->nr_Hp_BLCYC == pRPA->nuChi0Neig) && (pRPA->nc_Hp_BLCYC == pRPA->nuChi0Neig)) {
            info = LAPACKE_dsygvd(LAPACK_COL_MAJOR, 1, 'V', 'U', pRPA->nuChi0Neig, 
                pRPA->Hp, pRPA->nuChi0Neig, pRPA->Mp, pRPA->nuChi0Neig, pRPA->RRnuChi0Eigs);
            printf("global rank %d, the first 3 eigenvalues are %.5E, %.5E, %.5E\n", rankWorld, pRPA->RRnuChi0Eigs[0], pRPA->RRnuChi0Eigs[1], pRPA->RRnuChi0Eigs[2]);
            int nuChi0EigsCommRank;
            MPI_Comm_rank(pRPA->nuChi0Eigscomm, &nuChi0EigsCommRank);
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
        pdgemr2d_(&pRPA->nuChi0Neig, &pRPA->nuChi0Neig, pRPA->Hp, &ONE, &ONE, 
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
        int rank, nproc;
        MPI_Comm_rank(blacsComm, &rank);
        MPI_Comm_size(blacsComm, &nproc);

        int ONE = 1, il = 1, iu = 1, *ifail, info, N, M, NZ;
        double vl = 0.0, vu = 0.0, abstol, orfac;

        ifail = (int *)malloc(pRPA->nuChi0Neig * sizeof(int));
        N = pRPA->nuChi0Neig;
        orfac = 0.0;
        #ifdef DEBUG
        if(!rank) printf("rank = %d, orfac = %.3e\n", rank, orfac);
        #endif
        // this setting yields the most orthogonal eigenvectors
        // abstol = pdlamch_(&pSPARC->ictxt_blacs_topo, "U");
        abstol = -1.0;
        pdsygvx_subcomm_ (
                    &ONE, "V", "A", "U", &N, pRPA->Hp, &ONE, &ONE, 
                    pRPA->desc_Hp_BLCYC, pRPA->Mp, &ONE, &ONE, 
                    pRPA->desc_Mp_BLCYC, &vl, &vu, &il, &iu, &abstol, 
                    &M, &NZ, pRPA->RRnuChi0Eigs, &orfac, pRPA->Q, 
                    &ONE, &ONE, pRPA->desc_Q_BLCYC, ifail, &info,
                    blacsComm, pRPA->eig_paral_subdims, blksz);

        if (!rank){
            FILE *eigfile = fopen("eig.txt", "a");
            for (int col = 0; col < pRPA->nuChi0Neig; col++) {
                fprintf(eigfile, "%12.9f ", pRPA->RRnuChi0Eigs[col]);
            }
            fprintf(eigfile, "\n");
            fclose(eigfile);
        }
        if (printFlag) {
            if (pRPA->nr_Q_BLCYC > 1){
                char Qfile[50];
                snprintf(Qfile, 50, "rank%dblacsComm%d_eigVec_Q.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
                FILE *eigVecfile = fopen(Qfile, "a");
                fprintf(eigVecfile, "\n");
                for (int row = 0; row < pRPA->nr_Q_BLCYC; row++) {
                    for (int col = 0; col < pRPA->nc_Q_BLCYC; col++) {
                        fprintf(eigVecfile, "%12.9f ", pRPA->Q[col*pRPA->nr_Hp_BLCYC + row]);
                    }
                    fprintf(eigVecfile, "\n");
                }
                fclose(eigVecfile);
            }
        }
    }
#ifdef DEBUG    
    double et = MPI_Wtime();
    if (rankWorld == 0) printf("global rank = %d, generalized_eigenproblem_solver_gamma used %.3lf ms\n", rankWorld, 1000.0 * (et - st));
#endif
// #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)

// #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}

void automem_pzgeevx_( 
    const char *balanc, const char *jobvl, const char *jobvr, const char *sense, 
    const int *n, double _Complex *a, const int *desca, double _Complex *w, 
    double _Complex *vl, const int *descvl, double _Complex *vr, const int *descvr, 
    int *ilo, int *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif

	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork;
	double _Complex *work;
	lwork = -1;
	work  = (double _Complex*)malloc(100 * sizeof(double _Complex));  

	//** first do a workspace query **//
#ifdef DEBUG    
    t1 = MPI_Wtime();
#endif
	pzgeevx_(balanc, jobvl, jobvr, sense, n, a, desca, w, 
     vl, descvl, vr, descvr, 
     ilo, ihi, scale, abnrm, rconde, rcondv, 
     work, &lwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, creal(work[0]), (t2 - t1)*1e3);
#endif

	int NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	work = realloc(work, lwork * sizeof(double _Complex));

	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pzgeevx_(balanc, jobvl, jobvr, sense, n, a, desca, w, 
     vl, descvl, vr, descvr, 
     ilo, ihi, scale, abnrm, rconde, rcondv, 
     work, &lwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving standard "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pdgeevx, nuChi0Neig = %d\n", grank, *n);
}
#endif

	free(work);
#endif // (USE_MKL or USE_SCALAPACK)	
}

void subspace_rotation_unify_eigVecs_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
    // #if defined(USE_MKL) || defined(USE_SCALAPACK)
#ifdef DEBUG
    double st = MPI_Wtime();
#endif
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    int ONE = 1;
    int DMndspe = DMnd * Nspinor_eig;

    double alpha = 1.0, beta = 0.0;
    double t1, t2;

    t1 = MPI_Wtime();
    double *Y_BLCYC = pRPA->Ys_BLCYC;
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
    if(!rankWorld) printf("global rank = %2d, Distributing orbital back into band + domain format took %.3f ms\n", rankWorld, (t2 - t1)*1e3);
#endif

#ifdef DEBUG
    double et = MPI_Wtime();
     if(!rankWorld) printf("global rank = %d, Subspace_Rotation used %.3lf ms\n\n", rankWorld, 1000.0 * (et - st));
#endif
    for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
        double vec2norm;
        Vector2Norm(rotatedEigVecs + i*DMndspe, DMndspe, &vec2norm, dmcomm);
        VectorScale(rotatedEigVecs + i*DMndspe, DMndspe, 1.0/vec2norm, dmcomm); // unify the length of \Delta V
    }
// #endif // (USE_MKL or USE_SCALAPACK)

    // if (printFlag) {
    //     int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    //     if (1){ // (pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)
    //         char afterFilterName[100];
    //         snprintf(afterFilterName, 100, "rank%d_nuChi0Eigscomm%d_deltaVs_afterCheFSI.txt", rank, pRPA->nuChi0EigscommIndex);
    //         FILE *outputYs = fopen(afterFilterName, "a");
    //         if (outputYs ==  NULL) {
    //             printf("error printing deltaVs_afterCheFSI\n");
    //             exit(EXIT_FAILURE);
    //         } else {
    //             for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                 for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
    //                     fprintf(outputYs, "%12.9f ", rotatedEigVecs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
    //                 }
    //                 fprintf(outputYs, "\n");
    //             }
    //             fprintf(outputYs, "\n");
    //         }
    //         fclose(outputYs);
    //     }
    // }
}

double evaluate_cheFSI_error_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int flagNoDmcomm) {
    if (flagNoDmcomm) return 0.0;
    int rank; // nproc_dmcomm_phi,
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double *eigVecs = pRPA->deltaVs;
    double *nuChi0EigVecs = pRPA->Ys;
    int Nd_d_dmcomm = pSPARC->Nd_d_dmcomm;

    nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, eigVecs, nuChi0EigVecs, pRPA->nNuChi0Eigscomm, flagNoDmcomm, 0);
    double *diagEigVals = (double*)calloc(sizeof(double), pRPA->nNuChi0Eigscomm * pRPA->nNuChi0Eigscomm);
    for (int vec = pRPA->nuChi0EigsStartIndex; vec < pRPA->nuChi0EigsEndIndex + 1; vec++) {
        int localVec = vec - pRPA->nuChi0EigsStartIndex;
        diagEigVals[localVec*pRPA->nNuChi0Eigscomm + localVec] = pRPA->RRnuChi0Eigs[vec];
    }
    double *eigValEigVecs = (double*)calloc(sizeof(double), Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nd_d_dmcomm, pRPA->nNuChi0Eigscomm, pRPA->nNuChi0Eigscomm,
                  1.0, pRPA->deltaVs, Nd_d_dmcomm,
                  diagEigVals, pRPA->nNuChi0Eigscomm, 0.0,
                  eigValEigVecs, Nd_d_dmcomm);
    
    double *errorVecs = (double*)calloc(sizeof(double), pRPA->nNuChi0Eigscomm); double sumError = 0.0; double globalSumError = 0.0;
    for (int vec = 0; vec < pRPA->nNuChi0Eigscomm; vec++) {
        for (int i = 0; i < Nd_d_dmcomm; i++) {
            double errorTerm = eigValEigVecs[vec*Nd_d_dmcomm + i] - nuChi0EigVecs[vec*Nd_d_dmcomm + i];
            errorVecs[vec] += errorTerm * errorTerm;
        }
        sumError += sqrt(errorVecs[vec]);
    }
    MPI_Allreduce(&sumError, &globalSumError, 1, MPI_DOUBLE, MPI_SUM, pRPA->nuChi0EigsBridgeComm);
    double eigValEigVecsNorm = 0.0;
    for (int i = 0; i < pRPA->nuChi0Neig; i++) {
        eigValEigVecsNorm += pRPA->RRnuChi0Eigs[i] * pRPA->RRnuChi0Eigs[i];
    }
    eigValEigVecsNorm = sqrt(eigValEigVecsNorm);
    printf("nuChi0Eigscomm %d, rank %d, 1st eig %.5E, 1st element of 1st eigVal %.5E, 1st errorVec %.3E, sumError %.3E, globalSumError %.3E\n", 
        pRPA->nuChi0EigscommIndex, rank, pRPA->RRnuChi0Eigs[0], pRPA->deltaVs[0], errorVecs[0], sumError, globalSumError);
    double error = globalSumError / ((double)pRPA->nuChi0Neig * eigValEigVecsNorm);
    free(diagEigVals);
    free(eigValEigVecs);
    free(errorVecs);
    return error;
}