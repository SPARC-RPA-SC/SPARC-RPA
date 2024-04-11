#ifndef NUCHI0VECROUTINES
#define NUCHI0VECROUTINES

#include "isddft.h"
#include "main.h"

/**
 * @brief   Calculate LHS of Sternheimer equation times a bunch of vectors in a matrix-free way.
 *          LHS at here is (K-S Hamiltonian - epsilon_n + Q, if (flagPQ), |psi_n><psi_n| - i omega)
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param omegaIndex the index of omega of the current nu chi0 operator
 * @param DVs input vectors to multiply nu chi0, in dmcomm, not in dmcomm_phi
 * @param nuChi0DVs (out) product of nu chi0 operator and input vectors
 * @param nuChi0EigsAmount amount of solved eigenvalues in this nuChi0Eigscomm, at here it is the amount of DV vectors
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param printFlag flag to print mid variables
 */
void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs, double *nuChi0DVs, int nuChi0EigsAmount, int flagNoDmcomm, int printFlag);

void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt, double _Complex *nuChi0DVs_kpt, int nuChi0EigsAmount, int flagNoDmcomm);
#endif