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

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "cheFSI.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverKpt_RPA.h"
#include "tools_RPA.h"

void chebyshev_filtering_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, double minEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag) {
    double maxEig = -minEig;
    double e = (maxEig - lambdaCutoff) / 2.0;
    double c = (lambdaCutoff + maxEig) / 2.0;
    double sigma = e / (c - minEig);
    double sigmaNew;
    double tau = 2.0 / sigma;
    int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    int totalLength = nuChi0EigsAmount * pSPARC->Nd_d;

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

void project_YT_nuChi0_Y_kpt(RPA_OBJ* pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int isGammaPoint, int printFlag) {
    if (dmcomm_phi == MPI_COMM_NULL) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int nproc_dmcomm_phi, rankWorld;
    MPI_Comm_size(dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2, t3, t4;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double _Complex alpha = 1.0, beta = 0.0;
    double _Complex *Y = pRPA->Ys_kpt_phi;
    double _Complex *HY = pRPA->deltaVs_kpt_phi;
    // allocate memory for block cyclic format of the wavefunction
    double _Complex *Y_BLCYC;
    t3 = MPI_Wtime();

    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        Y_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        assert(Y_BLCYC != NULL);
        // distribute orbitals into block cyclic format
        pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, Y, &ONE, &ONE, pRPA->desc_orbitals,
                  Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); 
    } else {
        Y_BLCYC = pRPA->Ys_kpt_phi;
    }
    t2 = MPI_Wtime();  
    #ifdef DEBUG  
    if(!rankWorld) 
        printf("rank = %2d, Distribute orbital to block cyclic format took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3);          
    #endif
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
#ifdef DEBUG    
        if (!rankWorld) printf("rank = %d, STARTING PZGEMM ...\n",rankWorld);
#endif    
        // perform matrix multiplication using ScaLAPACK routines
        pzgemm_("C", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
                Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC,
                Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &beta, pRPA->Mp_kpt, 
                &ONE, &ONE, pRPA->desc_Mp_BLCYC);
    } else {
#ifdef DEBUG    
        if (!rankWorld) printf("rank = %d, STARTING ZGEMM ...\n",rankWorld);
#endif 
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            &alpha, Y_BLCYC, DMndspe, Y_BLCYC, DMndspe, 
            &beta, pRPA->Mp_kpt, pRPA->nuChi0Neig
        );
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) 
        printf("rank = %2d, Psi'*Psi in block cyclic format in each blacscomm took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3); 
#endif
    t1 = MPI_Wtime();
    if (nproc_dmcomm_phi > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, pRPA->Mp_kpt, pRPA->nr_Mp_BLCYC*pRPA->nc_Mp_BLCYC, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, dmcomm_phi);
    }
    t2 = MPI_Wtime();
    t4 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("rank = %2d, Allreduce to sum Psi'*Psi over dmcomm took %.3f ms\n", 
                     rankWorld, (t2 - t1)*1e3); 
    if(!rankWorld) printf("rank = %2d, Distribute data + matrix mult took %.3f ms\n", 
                     rankWorld, (t4 - t3)*1e3);
#endif
    double _Complex *HY_BLCYC;
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // distribute HY
        HY_BLCYC = (double _Complex *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        assert(HY_BLCYC != NULL);
        pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, HY, &ONE, &ONE, 
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
        pzgemm_("C", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
            Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, HY_BLCYC, 
            &ONE, &ONE, pRPA->desc_orb_BLCYC, &beta, pRPA->Hp_kpt, &ONE, &ONE, 
            pRPA->desc_Hp_BLCYC);
    } else {
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            &alpha, Y_BLCYC, DMndspe, HY_BLCYC, DMndspe, 
            &beta, pRPA->Hp_kpt, pRPA->nuChi0Neig
        );
    }

    if (nproc_dmcomm_phi > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, pRPA->Hp_kpt, pRPA->nr_Hp_BLCYC*pRPA->nc_Hp_BLCYC, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, dmcomm_phi);
    }

    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, finding Y'*HY took %.3f ms\n",rankWorld,(t2-t1)*1e3); 
#endif
    if (size_blacscomm > 1) {
        free(Y_BLCYC);
        free(HY_BLCYC);
    }

#ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, project_nuChi0 used %.3lf ms\n", 1000.0 * (et - st)); 
#endif
// // #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}