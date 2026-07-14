# Clean-Room Algorithm Catalog

For each GPL project we cannot incorporate, we document the algorithms we will reimplement independently.

| Algorithm | Source Project | License | Our Implementation |
|-----------|---------------|---------|-------------------|
| MRT collision (D'Humières) | PyLBM, lbmpy | GPL | Phase 2 |
| Smagorinsky SGS model | OpenLB, waLBerla | GPL | Phase 3 |
| STL voxelization | OpenLB | GPL | Phase 4 |
| Immersed boundary method | LUMA | Apache-2.0 | Phase 4 (can reference directly) |
| Block-structured AMR | MARBLES | Apache-2.0 | Phase 4 (can reference directly) |
| D3Q19/D3Q27 lattice | All | Various | Phase 2 (trivial constants) |
| Zou-He boundary conditions | OpenFOAM, Palabos | GPL | Phase 0.5 (well-known algorithm) |

## Rule

Read GPL source to understand the algorithm. Close the source. Implement from mathematical description and published papers. Document paper references for every implementation.
