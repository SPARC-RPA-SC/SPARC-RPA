/**
 * @file    parallelization_RPA.c
 * @brief   This file contains parallelization function for RPA calculation
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

void Setup_Comms_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    // phi domain communicator, or the communicator of the whole space, reuse pSPARC->dmcomm_phi
    // to compute Vxc (exact exchange?)
    // spin communicator and spin-bridge communicator also reuse pSPARC->spincomm and pSPARC->spin_bridge_comm
    // to compute Vxc with spin
    // kpt communicator should be composed in pRPA, decomposed from spin communicator
    // how to transfer psi (and potential, if necessary) from pSPARC to pRPA?
    // qpt-bridge communicator, which connects processors for the same qpt in different kpt comms, also decomposed from spin comm
    // to reduce Erpa(q)
    // qpt communicator decomposed from kpt communicator
    // to compute Hamiltonian
    // band communicator decomposed from every (kpt, qpt) unit, the relatively high level is for receiving psi from pSPARC->bandcomm 
    // to generate RHS of Sternheimer equation
    // omega communicator decomposed from every omega operator??
    // to add i\omega LHS of Sternheimer equation
    // domain communicator decomposed from every omega communicator
    // to decompose every single equation

    // if the system is only gamma point:
    // phi domain communicator, or the communicator of the whole space, reuse pSPARC->dmcomm_phi
    // to compute Vxc (exact exchange?)
    // spin communicator and spin-bridge communicator also reuse pSPARC->spincomm and pSPARC->spin_bridge_comm
    // to compute Vxc with spin
    // band communicator decomposed from every (kpt, qpt) unit, reuse pSPARC->bandcomm 
    // to generate RHS of Sternheimer equation
    // omega communicator decomposed from every omega operator??
    // to add i\omega LHS of Sternheimer equation
    // domain communicator decomposed from every omega communicator
    // to decompose every single equation
}