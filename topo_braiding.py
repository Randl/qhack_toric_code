import numpy as np
from toric_code import get_toric_code
from qiskit.circuit.library import C4XGate
from qiskit import transpile


sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, 1.0j], [-1.0j, 0]])
s0 = np.eye(2)


def em_braiding_phase(backend, x, y):
    tc = get_toric_code(x, y, classical_bit_count = 1, ancillas_count = 1)

    tc.circ.x(tc.regs[2][1])
    tc.circ.x(tc.regs[2][2])

    tc.circ.z(tc.regs[1][4])  # create m
    tc.circ.z(tc.regs[0][3])
    tc.circ.z(tc.regs[1][3])  # create m
    tc.circ.z(tc.regs[0][2])
    tc.circ.z(tc.regs[1][2])
    #tc.circ.z(tc.regs[3][4])  # create m

    tc.circ.h(tc.ancillas[0])


    cXXXX = np.kron(np.array([[1, 0],[0, 0]]), np.kron(sx, np.kron(sx, np.kron(sx,sx)))) + np.kron(np.array([[0, 0],[0, 1]]), np.kron(s0, np.kron(s0, np.kron(s0, s0))))

    #tc.circ.mct(control_qubits=[tc.ancillas[0]], target_qubit=[tc.ancillas[0], tc.regs[1][3], tc.regs[2][3], tc.regs[1][4], tc.regs[0][3]])
    tc.circ.unitary(cXXXX, [tc.regs[1][3], tc.regs[2][3], tc.regs[1][4], tc.regs[0][3], tc.ancillas[0]])
    #tc.circ.unitary(cXXXX, [tc.regs[2][2], tc.regs[3][3], tc.regs[4][2], tc.regs[3][2], tc.ancillas[0]])

    tc.circ.h(tc.ancillas[0])

    print(tc.circ)


    #for i in range(32):
    #    print(C4XGate().to_matrix()[i, :].real)



    tc.circ.measure(tc.ancillas[0], 0)
    Nshots = 10024
    job = backend.run(transpile(tc.circ, backend), shots=Nshots)
    result = job.result()
    counts = result.get_counts(tc.circ)


    return (counts['0'] - counts['1']) / Nshots, 0