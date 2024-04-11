#ifndef STERNEQ
#define STERNEQ

#include "isddft.h"
#include "main.h"

/**
 * @brief Collect all psis, epsilons and occupations in different bandcomms, which are used for composing initial guess of Sternheimer eq.s
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void collect_allXorb_allLambdas_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void collect_allXorb_allLambdas_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex);

/**
 * @brief Solve all Sternheimer eq.s for deltaVs (approx. eigenvectors of the nuChi0Eigscomm) and psis (K-S orbitals of the bandcomm) assigned to this processor
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param omegaIndex the index of omega of the current nu chi0 operator
 * @param nuChi0EigsAmount amount of solved eigenvalues in this nuChi0Eigscomm, at here it is the amount of DV vectors
 * @param DVs (input) deltaVs in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param printFlag flag to print mid variables
 */
void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, double *DVs, int printFlag);

void sternheimer_eq_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, double _Complex *DVs, int printFlag);

/**
 * @brief Solve a Sternheimer eq. for deltaVs and ONE psi
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param spn_i spin of the input orbital psi_n
 * @param epsilon (input) epsilon_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param omega (input) omega in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param flagPQ the flag to use linear operator P (|psi_n><psi_n| / dV) and Q (|psi_n><psi_n|)
 * @param flagCOCGinitial the flag to compose the initial guess of Sternheimer eq.s by eigenpairs got from K-S calculation
 * @param allXorb all orbitals from K-S calculation, for generating initial guess
 * @param allLambdas all eigenvalues of orbitals from K-S calculation, for generating initial guess
 * @param deltaPsisReal (output) the real part of solutions of this set of Sternheimer eq.s
 * @param deltaPsisImag (output) the imaginary part of solutions of this set of Sternheimer eq.s
 * @param deltaVs (input) deltaVs in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param psi (input) psi_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param bandWeight occupation*spin factor of orbital psi_n
 * @param deltaRhos (output) delta rho vectors from this orbital psi_n perturbed by deltaVs
 * @param nuChi0EigsAmounts the amounts of deltaVs
 * @param sternRelativeResTol the tolerance of MAXIMUM relative residuals. It can be replaced by Ferbenious norm to decrease iteration time
 * @param sternBlockSize the block size of RHS. All RHS [deltaV1 psi_n, deltaV2 psi_n, ..., deltaVn psi_n] are not solved together. They are solved block by block
 * @param printFlag flag to print mid variables
 * @param timeRecordName the file name to record the running time and iteration time of COCG
 */
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