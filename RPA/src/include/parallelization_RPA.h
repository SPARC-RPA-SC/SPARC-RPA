#ifndef PARALLEL_RPA
#define PARALLEL_RPA

#include "isddft.h"

void Setup_Comms_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput);

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex);

#endif