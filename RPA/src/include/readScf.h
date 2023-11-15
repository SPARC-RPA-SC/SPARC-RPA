#ifndef READSCFRPA
#define READSCFRPA

#include "isddft.h"

void read_scf(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex);

void read_eigValues_occs_rho(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex);

void read_orbitals(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex);

#endif