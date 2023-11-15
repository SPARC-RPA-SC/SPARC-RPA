#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "readScf.h"

void read_scf(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex) {
    read_eigValues_occs_rho(pSPARC, nuChi0EigscommIndex);
    read_orbitals(pSPARC, nuChi0EigscommIndex);
}

void read_eigValues_occs_rho(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex) {
    if (nuChi0EigscommIndex == -1) return;
}

void read_orbitals(SPARC_OBJ* pSPARC, int nuChi0EigscommIndex) {
    if (nuChi0EigscommIndex == -1) return; // not take part in computation
    if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex

    } else { // receiver of broadcast

    }
}