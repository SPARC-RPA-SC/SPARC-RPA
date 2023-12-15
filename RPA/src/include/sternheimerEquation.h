#ifndef STERNEQ
#define STERNEQ

#include "isddft.h"
#include "main.h"

double sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, int nuChi0EigsAmounts);

void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts);

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag);

double sternheimer_solver_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi, int nuChi0EigsAmounts);

void Sternheimer_lhs_kpt(SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *X, double _Complex *lhsX, int nuChi0EigsAmounts);

void set_initial_guess_deltaPsis_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *SternheimerRhs, double _Complex *deltaPsis);

#endif