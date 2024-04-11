#ifndef LINEARSOLVERRPA
#define LINEARSOLVERRPA

#include "isddft.h"
#include "main.h"

/**
 * @brief  Conjugate Orthogonal Conjugate Gradient (COCG) method for solving a set of Sternheimer equations with the same LHS.
 *         This linear solver does not accept vector distribution over many processors for now.
 * 
 * @param lhsfun the left-hand side of Sternheimer equation, a function mapping a vector to a vector
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param spn_i spin index of the band to be computed in this set of Sternheimer equation
 * @param psi the band vector of this set of Sternheimer equations, only one band!
 * @param epsilon the eigenvalue of this band psi
 * @param omega the omega in this set of Sternheimer equations
 * @param flagPQ the flag to apply linear operator Q(|psi><psi|) and P on both sides of Sternheimer equations
 * @param deltaPsisReal the real part of solution vectors
 * @param deltaPsiImag the imaginary part of solution vectors
 * @param SternheimerRhs the RHS vectors of Sternheimer equations
 * @param nuChi0EigsAmounts the amount of eigenpairs saved in this nuChi0Eigscomm, or the amount of RHS vectors in this Sternheimer equation set\
 * @param sternRelativeResTol the relative residual norm tolerance to judge convergence
 * @param maxIter the limit of iteration times
 * @param resNormRecords the record of residuals of all equations in all iterations
 * @param lhsTime the pointer recording the time spent on multiplying LHS by vectors
 * @param solveMuTime the pointer recording the time spent on solving the small linear equations
 * @param multipleTime the pointer recording the time spent on matrix multiplications
 */
int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, int, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int nuChi0EigsAmounts,
     double sternRelativeResTol, int maxIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime);

int kpt_solver(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *deltaPsis_kpt, 
     double _Complex *SternheimerRhs_kpt, int nuChi0EigsAmounts, int maxIter, double *resNormRecords);

/**
 * @brief  Judge whether the solution of COCG in this iteration has converged by comparing the relative residual of every equation
 *         with the relative residual tolerance.
 * 
 * @param ix the time of iteration
 * @param numVecs the amount of RHS vectors, or "nuChi0EigsAmounts" in function block_COCG
 * @param RHS2norm 2-norm of all RHS vectors
 * @param sternRelativeResTol the relative residual norm tolerance to judge convergence
 * @param resNormRecords the record of residuals of all equations in all iterations
 */
int judge_converge(int ix, int numVecs, const double RHS_frobnormsq, double sternRelativeResTol, const double *resNormRecords);

void AAR_kpt(
    SPARC_OBJ *pSPARC, 
    void (*res_fun)(SPARC_OBJ*,int,double,double,double,double,double _Complex*,double _Complex*,double _Complex*,MPI_Comm,double*),
    void (*precond_fun)(SPARC_OBJ *,int,double,double _Complex*,double _Complex*,MPI_Comm), double c, 
    int N, double qptx, double qpty, double qptz, double _Complex *x, double _Complex *b, double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
);

#endif