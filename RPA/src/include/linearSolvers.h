#ifndef LINEARSOLVERRPA
#define LINEARSOLVERRPA

#include "isddft.h"
#include "main.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double, double, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts, double tol, int maxIter, double *resNormRecords);

int kpt_solver(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *deltaPsis_kpt, 
     double _Complex *SternheimerRhs_kpt, int nuChi0EigsAmounts, int maxIter, double *resNormRecords);

void matrix_transpose(const double _Complex *M, int vecLength, int numVecs, double _Complex *MT);

void matrix_multiplication(const double _Complex *M, int MsizeRow, int MsizeCol, const double _Complex *x, int numVecs, double _Complex *Mx);

int judge_converge(int ix, int numVecs, const double *RHS2norm, double tol, const double *resNormRecords);

void divide_complex_vectors(double _Complex *complexVecs, double *realPart, double *imagPart, int length);

#endif