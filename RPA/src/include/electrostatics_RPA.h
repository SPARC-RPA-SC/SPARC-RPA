#ifndef ELECRPA
#define ELECRPA

#include "main.h"
#include "electrostatics_RPA.h"

/**
 * @brief Collect all delta rhos in different bandcomms to get delta rhos = chi0*deltaVs, these delta rho vectors are real vectors
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param deltaRhos delta rho vectors in different bandcomms
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0Eigscomm the communicator saving some deltaV vectors (eigvectors of nu chi0), every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 */
void collect_deltaRho_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm);

void collect_deltaRho_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm);

/**
 * @brief Compute deltaVs(new) = nu delta rhos = (nu chi0)*deltaVs by solving Laplace equation -1/4pi Lapla * delta Vs(new) = delta rho
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param deltaRhos delta rho vectors in different bandcomms
 * @param deltaVs (out) deltaV vectors, which is the product of nu chi0 operator and deltaVs from the last CheFSI iteration
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm to which this processor belongs
 * @param nuChi0Eigscomm the communicator saving some deltaV vectors (eigvectors of nu chi0), every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 */
void Calculate_sqrtNu_vecs_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, double *deltaVs, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

void Calculate_deltaRhoPotential_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, double _Complex *deltaVs_kpt, double qptx, double qpty, double qptz, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

/**
 * @brief Laplace equation solver, -Lapla * deltaVs(new) = 4pi*delta rho
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param deltaVs (out) deltaV vectors, which is the product of nu chi0 operator and deltaVs from the last CheFSI iteration
 * @param rhs at here it is delta rho
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm this processor belongs to
 */
void poissonSolver_gamma(SPARC_OBJ *pSPARC, double *deltaVs, double *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex);

void poissonSolver_kpt(SPARC_OBJ *pSPARC, double qptx, double qpty, double qptz, double _Complex *deltaVs_kpt, double _Complex *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex);

void poisson_residual_kpt(SPARC_OBJ *pSPARC, int N, double qptx, double qpty, double qptz, double c, double _Complex *x, double _Complex *b, double _Complex *r, MPI_Comm comm, double *time_info);

void Jacobi_preconditioner_kpt(SPARC_OBJ *pSPARC, int N, double c, double _Complex*r, double _Complex*f, MPI_Comm comm);

#endif