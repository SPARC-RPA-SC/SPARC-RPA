#ifndef PARALLEL_SPARC
#define PARALLEL_SPARC

#include "isddft.h"

void Setup_Comms_SPARC(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif