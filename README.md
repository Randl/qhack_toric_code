# Realizing topologically ordered states on a quantum processor

This repo contains reimplementaton of the paper
"[Realizing topologically ordered states on a quantum processor](https://arxiv.org/abs/2104.01180)" by Satzinger et al.
in Qiskit.

The systems of size up to 5x7 (31 qubit+ancilla) can be simulated on the classical computers,
reproducing the results of the 
## GS preparation
### Matching boundary conditions
For matching boundary condition, boundary plaquettes are all of the same type and the ground state is unique.
We implement linear algorithm for preparing ground state of the toric code proposed in the paper.
### Mixed boundary conditions
For mixed boundary condition, boundary plaquettes are of different types, and there is ground state degeneracy,
which allows us to encode logical qubits in the system state.
Not implemented yet.

## Entropy and topological entropy
We implement the measurement of the second Rényi entropy as described in paper, and use it
to calculate topological entropy, acquiring non-trivial value for various subsystems.

For 2x2 and 2x3 subsystem it is possible to perform determenistic calculation, while for 
3x3 system only randomized calculation is feasible.

## Braiding
There are 4 particle types in toric code: `e`, `m`, `psi` and `1`, for total of 6 possible 
mutual statistics and 3 exchange statistic. Out of those 4 are non-trivial -- `em`, `epsi`,
`mpsi` and `psipsi`, resulting in phase of π.

We implement the required operators (without optiimization), and demonstrate part of the braidings
and exchanges.
## Logical qubit.
Not implemented yet.