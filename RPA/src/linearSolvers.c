#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "linearSolvers.h"
#include "tools_RPA.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double, double, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts, double tol, int maxIter, double *resNormRecords) {
    
    int DMnd = pSPARC->Nd_d_dmcomm;
    int rhsLength = DMnd * nuChi0EigsAmounts;
    int rhoLength = nuChi0EigsAmounts * nuChi0EigsAmounts;
    double *RHS2norm = (double*)calloc(sizeof(double), nuChi0EigsAmounts);
    for (int vecIndex = 0; vecIndex < nuChi0EigsAmounts; vecIndex++) {
         Vector2Norm_complex(SternheimerRhs + vecIndex*DMnd, DMnd, &(RHS2norm[vecIndex]), pSPARC->dmcomm);
    }

    double _Complex *LHSx = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    double _Complex *V = (double _Complex*)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    lhsfun(pSPARC, spn_i, epsilon, omega, deltaPsisReal, deltaPsisImag, LHSx, nuChi0EigsAmounts);
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
    double Nalpha = 1.0; double Nbeta = 0.0;
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &Nalpha, V, DMnd,
                  W, DMnd, &Nbeta,
                  rho, nuChi0EigsAmounts);
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
        if (judge_converge(ix, nuChi0EigsAmounts, RHS2norm, tol, resNormRecords)) {
            break;
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &Nalpha, P, DMnd,
                  beta, nuChi0EigsAmounts, &Nbeta,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            P[i] = W[i] + midVariable[i]; // P = W + P*beta;
        }
        divide_complex_vectors(P, dividedReal, dividedImag, rhsLength);
        lhsfun(pSPARC, spn_i, epsilon, omega, dividedReal, dividedImag, U, nuChi0EigsAmounts); // U = Afun(P);
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &Nalpha, U, DMnd,
                  P, DMnd, &Nbeta,
                  mu, nuChi0EigsAmounts); // mu = U.' * P;
        MKL_INT ipiv[nuChi0EigsAmounts];
        memcpy(alpha, rho, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        memcpy(lapackMu, mu, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', nuChi0EigsAmounts, nuChi0EigsAmounts, lapackMu, nuChi0EigsAmounts, ipiv, alpha, nuChi0EigsAmounts ); // alpha = mu \ rho;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &Nalpha, P, DMnd,
                  alpha, nuChi0EigsAmounts, &Nbeta,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            deltaPsisReal[i] += creal(midVariable[i]);
            deltaPsisImag[i] += cimag(midVariable[i]); // X = X + P*alpha;
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, nuChi0EigsAmounts,
                  &Nalpha, U, DMnd,
                  alpha, nuChi0EigsAmounts, &Nbeta,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            V[i] -= midVariable[i]; // V = V - U*alpha;
        }
        memcpy(W, V, sizeof(double _Complex)*rhsLength); // W = V;
        for (int vecIndex = 0; vecIndex < nuChi0EigsAmounts; vecIndex++) {
            Vector2Norm_complex(V + vecIndex*DMnd, DMnd, &(resNormRecords[(ix + 1)*nuChi0EigsAmounts + vecIndex]), pSPARC->dmcomm);
        }
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nuChi0EigsAmounts, nuChi0EigsAmounts, DMnd,
                  &Nalpha, V, DMnd,
                  W, DMnd, &Nbeta,
                  rhoNew, nuChi0EigsAmounts); // rho_new = V.' * W;
        memcpy(beta, rhoNew, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        memcpy(lapackRho, rho, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts);
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', nuChi0EigsAmounts, nuChi0EigsAmounts, lapackRho, nuChi0EigsAmounts, ipiv, beta, nuChi0EigsAmounts ); // beta = rho \ rho_new;
        memcpy(rho, rhoNew, sizeof(double _Complex)*nuChi0EigsAmounts*nuChi0EigsAmounts); // rho = rho_new;
    }
    printf("block COCG iterated for %d times; residual %.6E", ix, resNormRecords[ix*nuChi0EigsAmounts]);
    if (ix == maxIter) {
        printf("It terminated without converging to the desired tolerance.\n");
    } else {
        printf("\n");
    }

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

int judge_converge(int ix, int numVecs, const double *RHS2norm, double tol, const double *resNormRecords) {
    int judge = 1;
    for (int col = 0; col < numVecs; col++) {
        if (resNormRecords[numVecs*ix + col] > RHS2norm[col]*tol) {
            judge = 0;
            break;
        }
    }
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