#ifndef PARALLEL_RPA
#define PARALLEL_RPA

#include "isddft.h"

void Setup_Comms_RPA(RPA_OBJ *pRPA, int Nspin, int Nkpts, int Nstates);

void dims_divide_Eigs(int nuChi0Neig, int Nspin, int Nkpts, int Nstates, int *npnuChi0Neig, int *npspin, int *npkpt, int *npband);

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput);

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex);

#endif