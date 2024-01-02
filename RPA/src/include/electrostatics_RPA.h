#ifndef ELECRPA
#define ELECRPA

#include "main.h"
#include "electrostatics_RPA.h"

void collect_transfer_deltaRho_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, double *deltaRhos_phi, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm);

void collect_transfer_deltaRho_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, double _Complex *deltaRhos_kpt_phi, int nuChi0EigsAmount, int printFlag, MPI_Comm nuChi0Eigscomm);

void transfer_deltaRho(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *rho_send, double *rho_recv);

void transfer_deltaRho_kpt(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double _Complex *rho_send, double _Complex *rho_recv);

void Calculate_deltaRhoPotential_gamma(SPARC_OBJ *pSPARC, double *deltaRhos_phi, double *deltaVs_phi, int nuChi0EigsAmount, int printFlag, double *deltaVsForPrint, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

void Calculate_deltaRhoPotential_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt_phi, double _Complex *deltaVs_kpt_phi, double qptx, double qpty, double qptz, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

void poissonSolver_gamma(SPARC_OBJ *pSPARC, double *deltaVs_phi, double *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex);

void poissonSolver_kpt(SPARC_OBJ *pSPARC, double qptx, double qpty, double qptz, double _Complex *deltaVs_kpt_phi, double _Complex *rhs, int nuChi0EigsAmounts, int nuChi0EigscommIndex);

void poisson_residual_kpt(SPARC_OBJ *pSPARC, int N, double qptx, double qpty, double qptz, double c, double _Complex *x, double _Complex *b, double _Complex *r, MPI_Comm comm, double *time_info);

void Jacobi_preconditioner_kpt(SPARC_OBJ *pSPARC, int N, double c, double _Complex*r, double _Complex*f, MPI_Comm comm);

#endif