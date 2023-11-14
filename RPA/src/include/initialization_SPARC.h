#ifndef INITIALIZATION_RPA
#define INITIALIZATION_RPA

#include "isddft.h"

void Initialize_SPARC_before_SetComm(SPARC_OBJ *pSPARC, int argc, char *argv[]);

void Initialize_SPARC_SetComm_after(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif