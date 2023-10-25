#ifndef INITIALRPA 
#define INITIALRPA

#include "isddft.h"

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]);

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI);

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, int Nstates, int Nd);

void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input);

void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input);

void set_omegas(double *omega, double *omega01, double *omegaWts, int Nomega);

void write_settings(RPA_OBJ *pRPA);

#endif