#ifndef GROUNDRPA 
#define GROUNDRPA 

#include "isddft.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif