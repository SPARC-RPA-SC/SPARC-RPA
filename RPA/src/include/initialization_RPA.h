#include "isddft.h"

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]);

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, SPARC_OBJ *pSPARC);

void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input);

void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input);

void write_settings(RPA_OBJ *pRPA);

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI);