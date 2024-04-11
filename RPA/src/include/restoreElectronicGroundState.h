#ifndef GROUNDRPA 
#define GROUNDRPA 

#include "isddft.h"

/**
 * @brief restore all needed information and settings from the previous K-S calculation in the SPARC object
 *        K-S calculation has time-reverse symmetry (symmetrized k-points), but RPA needs to use complete k-point list.
 *        So necessary to find the mapping from the coord of k+q in complete k-point list to the index of it in symmetrized k-point list
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param nuChi0Eigscomm the nu chi0 eigs communicator having this SPARC object
 * @param nuChi0EigsBridgeComm the communicator to connect all processors having the same rank in their nu chi0 eigs communicators
 * @param nuChi0EigscommIndex the index of the nu chi0 eigs communicator
 * @param rank0nuChi0EigscommInWorld global rank of the 0th processor of the nu chi0 eigs communicator
 * @param symk1 the 1st reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 2nd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 3rd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param kPqSymList the list mapping the coord of k-point (from complete k-point list) + q-point (from symmetrized k-point list without shift)
 * @param Nkpts_sym the total number of k-points in symmetrized k-point list to its corresponding symmetrized k-point index.
 */
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