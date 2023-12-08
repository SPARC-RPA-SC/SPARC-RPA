#ifndef GROUNDRPA 
#define GROUNDRPA 

#include "isddft.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld,
     double *symk1, double *symk2, double *symk3, int **kPqSymList, int Nkpts_sym);

void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld, double *symk1, double *symk2, double *symk3, int **kPqSymList);

void read_orbitals_distributed_gamma_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm);

void read_orbitals_distributed_real_RPA(SPARC_OBJ *pSPARC, int band, int spin, MPI_Datatype domainSubarray, double *readXorb);

void read_orbitals_distributed_kpt_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *symk1, double *symk2, double *symk3, int **kPqSymList);

void read_orbitals_distributed_complex_RPA(SPARC_OBJ *pSPARC, int flagSymKpt, int kpt, double symk1, double symk2, double symk3, int band, int spin, MPI_Datatype domainSubarray, double _Complex *readXorb_kpt);

void restore_electronDensity(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

void restore_eigval_occ(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int Nkpts_sym, int **kPqSymList);

void read_eigval_occ(char *inputEigsFnames, int Nspin, int Nkpts_sym, int Nstates, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym);

void find_eigval_occ_spin_kpts(SPARC_OBJ *pSPARC, int Nkpts_sym, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym, int **kPqSymList);

void Transfer_Veff_loc_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *Veff_phi_domain, double *Veff_psi_domain);

void Transfer_Veff_loc_RPA_kpt(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double _Complex *Veff_phi_domain, double _Complex *Veff_psi_domain);
#endif