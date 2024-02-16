#ifndef EIGGAMMARPA
#define EIGGAMMARPA

#include "isddft.h"
#include "main.h"

/**
 * @brief Make Chebyshev filering on operator nu chi0 (omega)
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param omegaIndex the index of omega of the current nu chi0 operator
 * @param minEig a parameter of Chebyshev filter, the minimum eigenvalue of operator nu chi0, found by compute_ErpaTerm
 * @param maxEig a parameter of Chebyshev filter, not the real maximum eigenvalue of nu chi0
 * @param lambdaCutoff a parameter of Chebyshev filter, near the largest eigenvalue to be solved
 * @param chebyshevDegree the degree of Chebyshev polynomial used in the filter
 * @param printFlag flag to print mid variables
 */
void chebyshev_filtering_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, double minEig, double maxEig, double lambdaCutoff, int chebyshevDegree, int flagNoDmcomm, int printFlag);

/**
 * @brief Multiply the filtered vectors Y^T by itself Y
 *
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm_phi the phi domain in pSPARC, saving filtered vectors Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in phi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param printFlag flag to print mid variables
 */
void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

/**
 * @brief Orthogonalize filtered vectors Y, it is necessary if the eigenproblem is solved in parallel: if ScaLapack
 * is selected to solve eigenpairs of (YT * nu Chi0 * Y), since ScaLapack does not support nonsymmetric generalized
 * eigenvalue problems, we have to solve nonsymmetric eigenvalue problems AX = X Lambda
 *
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param pRPA pointer to RPA_OBJ
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in phi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param printFlag flag to print mid variables
 */
void Y_orth_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int DMnd, int Nspinor_eig, int printFlag);

/**
 * @brief Project operator nu chi0 into space spanned by Y, Y^T * nu chi0 * Y
 *
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm_phi the phi domain in pSPARC, saving filtered vectors Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in phi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param printFlag flag to print mid variables
 */
void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, int printFlag);

/**
 * @brief Solve the eigenpairs of (Y^T * nu chi0 * Y). If it is solved in serial, it is a generalized eigenproblem (Y is not orthogonailized);
 * if it is solved in parallel, it it an ordinal eigenproblem (Y is orthogonalized)
 *
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm_phi the phi domain in pSPARC, saving filtered vectors Y
 * @param blacsComm blacsComm in pRPA (not pSPARC!), connecting phi domains of all nuChi0Eigscomms
 * @param blksz block size for distributing matrix (Y^T * nu chi0 * Y), used if it is solved in parallel
 * @param signImag the sign to inform that some eigenvalues have large imagination part
 * @param printFlag flag to print mid variables
 */
void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, MPI_Comm blacsComm, int blksz, int *signImag, int printFlag);

void automem_pdgeevx_( 
    const char *balanc, const char *jobvl, const char *jobvr, const char *sense, 
    const int *n, double *a, const int *desca, double *wr, double *wi, 
    double *vl, const int *descvl, double *vr, const int *descvr, 
    int *ilo, int *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, int *info);

/**
 * @brief Call pzgeevx function to solve nonsymmetric eigenvalue problems AX = X Lambda, but matrix 
 * (YT * nu Chi0 * Y) is redistribute in pRPA->blacscomm before calling (not solved in text pRPA->ictxt_blacs_topo). 
 * pdgeevx has bugs in solving eigenvectors. The input variables are input variables of MKL routine pzgeevx
 * @param a distributed matrix, It was (Y^T * nu chi0 * Y) in text pRPA->ictxt_blacs_topo (currently it is not)
 * @param w (out) solved eigenvalues
 * @param vr (out) solved eigenvectors, distributed. It is going to be Q in text pRPA->ictxt_blacs_topo (currently it is not)
 *
 */
void automem_pzgeevx_( 
    const char *balanc, const char *jobvl, const char *jobvr, const char *sense, 
    const int *n, double _Complex *a, const int *desca, double _Complex *w, 
    double _Complex *vl, const int *descvl, double _Complex *vr, const int *descvr, 
    int *ilo, int *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, int *info);

/**
 * @brief Restore eigenvectors Q from space spanned by Y back to real space, Y*Q
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm_phi the phi domain in pSPARC, saving filtered vectors Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in phi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param rotatedEigVecs (out) Y*Q
 * @param printFlag flag to print mid variables
 *
 */
void subspace_rotation_unify_eigVecs_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm_phi, int DMnd, int Nspinor_eig, double *rotatedEigVecs, int printFlag);
#endif