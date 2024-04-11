#ifndef STERNEQ
#define STERNEQ

#include "isddft.h"
#include "main.h"

void collect_allXorb_allLambdas_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void collect_allXorb_allLambdas_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex);

void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, double *DVs, int printFlag);

void sternheimer_eq_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, double _Complex *DVs, int printFlag);

void sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, int flagCOCGinitial, double *allXorb, double *allLambdas,
                                double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, double bandWeight, double *deltaRhos, int nuChi0EigsAmounts,
                                double sternRelativeResTol, int inputSternBlockSize, int printFlag, char *timeRecordname);

void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts);

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, int flagCOCGinitial, double *allXorb, double *allLambdas,
    double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag, int nuChi0EigsAmounts);

double sternheimer_solver_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *deltaPsis_kpt, double _Complex *deltaVs_kpt, double _Complex *psi, double occ, double _Complex *deltaRhos_kpt, int nuChi0EigsAmounts);

void Sternheimer_lhs_kpt(SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *X, double _Complex *lhsX, int nuChi0EigsAmounts);

void set_initial_guess_deltaPsis_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, int kMq, double epsilon, double omega, double _Complex *SternheimerRhs, double _Complex *deltaPsis);

#endif