#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>

#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "linearSolvers.h"
#include "tools_RPA.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, int, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts,
     double sternRelativeResTol, int maxIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime) {
    double lhsTimeRecord = 0.0; double solveMuTimeRecord = 0.0; double multipleTimeRecord = 0.0;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int rhsLength = DMnd * nuChi0EigsAmounts;
    int rhoLength = nuChi0EigsAmounts * nuChi0EigsAmounts;
    double *RHS2norm = (double*)calloc(sizeof(double), nuChi0EigsAmounts);
    double RHS_frobnormsq = 0;
    for (int vecIndex = 0; vecIndex < nuChi0EigsAmounts; vecIndex++) {
         Vector2Norm_complex(SternheimerRhs + vecIndex*DMnd, DMnd, &(RHS2norm[vecIndex]), pSPARC->dmcomm);
         RHS_frobnormsq += RHS2norm[vecIndex]*RHS2norm[vecIndex];
    }

    double _Complex *LHSx = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double _Complex *V = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    #ifdef DEBUG
    double t1 = MPI_Wtime();
    #endif
    lhsfun(pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisReal, deltaPsisImag, LHSx, nuChi0EigsAmounts);
    #ifdef DEBUG
    double t2 = MPI_Wtime();
    lhsTimeRecord += (t2 - t1);
    #endif
    for (int rhsIndex = 0; rhsIndex < rhsLength; rhsIndex++) {
        V[rhsIndex] = SternheimerRhs[rhsIndex] - LHSx[rhsIndex];
    }
    for (int vecIndex = 0; vecIndex < nuChi0EigsAmounts; vecIndex++) {
        Vector2Norm_complex(V + vecIndex*DMnd, DMnd, &(resNormRecords[vecIndex]), pSPARC->dmcomm);
    }
    double _Complex *W = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    memcpy(W, V, sizeof(double _Complex) * rhsLength);
    double _Complex *rho = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *lapackRho = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    // Reminder: if there is domain parallelization, then it is necessary to call pdgemm_ function in ScaLapack with blacs
    // to make the distributed matrix multiplication
    double _Complex one = 1.0; double _Complex zero = 0.0; double _Complex minus1 = -1.0;
    #ifdef DEBUG
    double t3 = MPI_Wtime();
    #endif
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &one, V, DMnd,
                  W, DMnd, &zero,
                  rho, nuChi0EigsAmounts);
    #ifdef DEBUG
    double t4 = MPI_Wtime();
    multipleTimeRecord += (t4 - t3);
    #endif
    double _Complex *P = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double _Complex *beta = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *U = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double _Complex *mu = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *lapackMu = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *alpha = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *rhoNew = (double _Complex*)calloc(sizeof(double _Complex), nuChi0EigsAmounts*nuChi0EigsAmounts);
    double _Complex *midVariable = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double *dividedReal = (double*)calloc(sizeof(double), DMnd*nuChi0EigsAmounts);
    double *dividedImag = (double*)calloc(sizeof(double), DMnd*nuChi0EigsAmounts);
    MKL_INT info;
    int ix;
    for (ix = 0; ix < maxIter; ix++) {
        if (judge_converge(ix, nuChi0EigsAmounts, RHS_frobnormsq, sternRelativeResTol, resNormRecords)) {
            break;
        }
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &one, P, DMnd,
                  beta, nuChi0EigsAmounts, &zero,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            P[i] = W[i] + midVariable[i]; // P = W + P*beta;
        }
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        divide_complex_vectors(P, dividedReal, dividedImag, rhsLength);
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        lhsfun(pSPARC, spn_i, psi, epsilon, omega, flagPQ, dividedReal, dividedImag, U, nuChi0EigsAmounts); // U = Afun(P);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        lhsTimeRecord += (t2 - t1);
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &one, U, DMnd,
                  P, DMnd, &zero,
                  mu, nuChi0EigsAmounts); // mu = U.' * P;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        MKL_INT ipiv[nuChi0EigsAmounts];
        memcpy(alpha, rho, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        memcpy(lapackMu, mu, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        #ifdef DEBUG
        double t5 = MPI_Wtime();
        #endif
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', nuChi0EigsAmounts, nuChi0EigsAmounts, lapackMu, nuChi0EigsAmounts, ipiv, alpha, nuChi0EigsAmounts ); // alpha = mu \ rho;
        #ifdef DEBUG
        double t6 = MPI_Wtime();
        solveMuTimeRecord += (t6 - t5);
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &one, P, DMnd,
                  alpha, nuChi0EigsAmounts, &zero,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            deltaPsisReal[i] += creal(midVariable[i]);
            deltaPsisImag[i] += cimag(midVariable[i]); // X = X + P*alpha;
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &minus1, U, DMnd,
                  alpha, nuChi0EigsAmounts, &one,
                  V, DMnd); // V = V - U*alpha;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        memcpy(W, V, sizeof(double _Complex)*rhsLength); // W = V;
        for (int vecIndex = 0; vecIndex < nuChi0EigsAmounts; vecIndex++) {
            Vector2Norm_complex(V + vecIndex*DMnd, DMnd, &(resNormRecords[(ix + 1)*nuChi0EigsAmounts + vecIndex]), pSPARC->dmcomm);
        }
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &one, V, DMnd,
                  W, DMnd, &zero,
                  rhoNew, nuChi0EigsAmounts); // rho_new = V.' * W;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        memcpy(beta, rhoNew, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        memcpy(lapackRho, rho, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        #ifdef DEBUG
        t5 = MPI_Wtime();
        #endif
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', nuChi0EigsAmounts, nuChi0EigsAmounts, lapackRho, nuChi0EigsAmounts, ipiv, beta, nuChi0EigsAmounts ); // beta = rho \ rho_new;
        #ifdef DEBUG
        t6 = MPI_Wtime();
        solveMuTimeRecord += (t6 - t5);
        #endif
        memcpy(rho, rhoNew, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts); // rho = rho_new;
    }
    *lhsTime = lhsTimeRecord; *solveMuTime = solveMuTimeRecord; *multipleTime = multipleTimeRecord;
    /*
    printf("block COCG iterated for %d times; residual %.6E", ix, resNormRecords[ix*nuChi0EigsAmounts]);
    if (ix == maxIter) {
        printf("It terminated without converging to the desired tolerance.\n");
    } else {
        printf("\n");
    }
    */

    free(RHS2norm);
    free(LHSx);
    free(V);
    free(W);
    free(rho);
    free(lapackRho);
    free(P);
    free(beta);
    free(U);
    free(mu);
    free(lapackMu);
    free(alpha);
    free(rhoNew);
    free(midVariable);
    free(dividedReal);
    free(dividedImag);
    return ix;
}

int kpt_solver(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *deltaPsis_kpt, 
     double _Complex *SternheimerRhs_kpt, int nuChi0EigsAmounts, int maxIter, double *resNormRecords) {
    // solve the two Sternheimer eq.s (-i\omega and +i\omega) together

    return 0;
}

int judge_converge(int ix, int numVecs, const double RHS_frobnormsq, double sternRelativeResTol, const double *resNormRecords) {
    // int judge = 1;
    double resid_frobnormsq = 0;
    for (int col = 0; col < numVecs; col++)
    {
        resid_frobnormsq += resNormRecords[numVecs*ix + col]*resNormRecords[numVecs*ix + col];
    }
    int judge = (resid_frobnormsq / RHS_frobnormsq) > sternRelativeResTol*sternRelativeResTol ? 0 : 1;
/*
    for (int col = 0; col < numVecs; col++) {
        if (resNormRecords[numVecs*ix + col] > RHS2norm[col]*sternRelativeResTol) {
            judge = 0;
            break;
        }
    }
*/
    // double sum = 0.0;
    // for (int i = 0; i < numVecs; i++) {
    //     sum += resNormRecords[ix*numVecs + i];
    // }
    // int judge = (sum / sqrt((double)numVecs)) > tol ? 0 : 1;
    return judge;
}

void AAR_kpt(
    SPARC_OBJ *pSPARC, 
    void (*res_fun)(SPARC_OBJ*,int,double,double,double,double,double _Complex*,double _Complex*,double _Complex*,MPI_Comm,double*),  
    void (*precond_fun)(SPARC_OBJ *,int,double,double _Complex*,double _Complex*,MPI_Comm), double c, 
    int N, double qptx, double qpty, double qptz, double _Complex *x, double _Complex *b, double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
) {

}
