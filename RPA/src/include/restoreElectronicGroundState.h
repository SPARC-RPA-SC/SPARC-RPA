#ifndef GROUNDRPA 
#define GROUNDRPA 

#include "isddft.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

void restore_electronDensity(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif