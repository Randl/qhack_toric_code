import numpy as np
from qiskit import transpile

from toric_code import get_toric_code
from toric_code_matching import is_inside_matching

sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, 1.0j], [-1.0j, 0]])
s0 = np.eye(2)


def create_e_particles(tc, string):
    for x, y in string:
        tc.circ.x(tc.regs[x][y])


def create_m_particles(tc, string):
    for x, y in string:
        tc.circ.z(tc.regs[x][y])


def apply_cxxxx_on_square(tc, upper_corner):
    x, y = upper_corner
    if x % 2 == 0:
        locs = [(x, y), (x + 1, y), (x + 2, y), (x + 1, y + 1)]
    else:
        locs = [(x, y), (x + 1, y - 1), (x + 2, y), (x + 1, y)]

    if not all([is_inside_matching((tc.x, tc.y), l) for l in locs]):
        return
    tc.circ.mct(control_qubits=[tc.ancillas[0]], target_qubit=[tc.regs[l[0]][l[1]] for l in locs])


def apply_czzz_on_square(tc, upper_corner):
    cZZZZ = np.kron(np.array([[1, 0], [0, 0]]), np.kron(sz, np.kron(sz, np.kron(sz, sz)))) + \
            np.kron(np.array([[0, 0], [0, 1]]), np.kron(s0, np.kron(s0, np.kron(s0, s0))))
    x, y = upper_corner
    if x % 2 == 0:
        locs = [(x, y), (x + 1, y), (x + 2, y), (x + 1, y + 1)]
    else:
        locs = [(x, y), (x + 1, y - 1), (x + 2, y), (x + 1, y)]

    if not all([is_inside_matching((tc.x, tc.y), l) for l in locs]):
        return

    tc.circ.unitary(cZZZZ, [tc.regs[l[0]][l[1]] for l in locs] + [tc.ancillas[0]])


def apply_cxxyyzz_on_rectangle(tc, upper_corner, left=True):
    cXXYYZZ = np.kron(np.array([[1, 0], [0, 0]]), np.kron(sx, np.kron(sx, np.kron(sy, np.kron(sy, np.kron(sz, sz)))))) + \
              np.kron(np.array([[0, 0], [0, 1]]), np.kron(s0, np.kron(s0, np.kron(s0, np.kron(s0, np.kron(s0, s0))))))

    x, y = upper_corner
    if left:
        if x % 2 == 0:
            locs = [(x, y), (x + 1, y + 1), (x + 1, y), (x + 2, y), (x + 2, y - 1), (x + 3, y)]
        else:
            locs = [(x, y), (x + 1, y), (x + 1, y - 1), (x + 2, y), (x + 1, y - 1), (x + 2, y - 1)]
    else:
        if x % 2 == 0:
            locs = [(x, y), (x + 1, y), (x + 1, y + 1), (x + 2, y), (x + 2, y + 1), (x + 3, y + 1)]
        else:
            locs = [(x, y), (x + 1, y - 1), (x + 1, y), (x + 2, y), (x + 2, y + 1), (x + 3, y)]

    if not all([is_inside_matching((tc.x, tc.y), l) for l in locs]):
        return
    tc.circ.unitary(cXXYYZZ, [tc.regs[l[0]][l[1]] for l in locs[::-1]] + [tc.ancillas[0]])


def em_braiding_phase(backend, x, y):
    tc = get_toric_code(x, y, classical_bit_count=1, ancillas_count=1)

    x_string = [(2, 1), (2, 2)]
    create_e_particles(tc, x_string)

    z_string = [(1, 4), (0, 3), (1, 3)]
    create_m_particles(tc, z_string)

    tc.circ.h(tc.ancillas[0])

    apply_cxxxx_on_square(tc, (0, 3))
    tc.circ.h(tc.ancillas[0])

    tc.circ.measure(tc.ancillas[0], 0)
    Nshots = 10000
    job = backend.run(transpile(tc.circ, backend), shots=Nshots)
    result = job.result()
    counts = result.get_counts(tc.circ)

    if '0' not in counts:
        counts['0'] = 0
    if '1' not in counts:
        counts['1'] = 0

    cos_theta = (counts['0'] - counts['1']) / Nshots
    return cos_theta, 0
