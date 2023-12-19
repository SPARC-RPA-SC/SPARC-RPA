#ifndef ELECRPA
#define ELECRPA

#include "main.h"
#include "electrostatics_RPA.h"

void collect_transfer_deltaRho(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void Calculate_deltaRhoPotential(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex);

void transfer_deltaRho(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *rho_send, double *rho_recv);

void transfer_deltaRho_kpt(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double _Complex *rho_send, double _Complex *rho_recv);

#endif