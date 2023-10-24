#ifndef PARALLEL_RPA
#define PARALLEL_RPA

#include "isddft.h"

void Setup_Comms_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void dims_divide_QptOmegaEigs(int nqpts_sym, int Nomega, int nuChi0Neig, int *npqpt, int *npomega, int *npnuChi0Neig);

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput);

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex);

#endif