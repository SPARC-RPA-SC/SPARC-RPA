#ifndef LINEARSOLVERRPA
#define LINEARSOLVERRPA

#include "isddft.h"
#include "main.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double *psi, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts, double tol, int maxIter, double *resNormRecords);

int kpt_solver(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *deltaPsis_kpt, 
     double _Complex *SternheimerRhs_kpt, int nuChi0EigsAmounts, int maxIter, double *resNormRecords);

int judge_converge(int ix, int numVecs, const double *RHS2norm, double tol, const double *resNormRecords);

void AAR_kpt(
    SPARC_OBJ *pSPARC, 
    void (*res_fun)(SPARC_OBJ*,int,double,double,double,double,double _Complex*,double _Complex*,double _Complex*,MPI_Comm,double*),
    void (*precond_fun)(SPARC_OBJ *,int,double,double _Complex*,double _Complex*,MPI_Comm), double c, 
    int N, double qptx, double qpty, double qptz, double _Complex *x, double _Complex *b, double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
);

#endif