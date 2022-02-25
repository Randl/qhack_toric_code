import itertools
import unittest

import numpy as np
from qiskit import Aer
from qiskit import transpile
from tqdm import tqdm

from topo_entropy import ABC_DIVISION_2x2
from topo_entropy import calculate_s_subsystems
from topo_entropy import get_all_2x2_non_corner, get_all_2x3_non_corner, get_all_3x3_non_corner
from toric_code import get_toric_code


def test_topo_entropy(backend, size, qubits, subsystems, expected_values, type='haar', cnt=1000):
    assert type in ('haar', 'pauli')

    x, y = size
    all_counts = []
    if type == 'haar':
        all_gates = range(cnt)
    elif type == 'pauli':
        all_gates = [''.join(x) for x in itertools.product('xyz', repeat=len(qubits))]
    for gates in tqdm(all_gates):
        tc = get_toric_code(x, y, len(qubits))

        if type == 'haar':
            tc.measure_haar(qubits)
        elif type == 'pauli':
            tc.measure_pauli(qubits, gates)

        job = backend.run(transpile(tc.circ, backend), shots=1024)
        result = job.result()
        counts = result.get_counts(tc.circ)
        all_counts.append(counts)
    calculated_values = calculate_s_subsystems(all_counts, subsystems)
    for expect, calc in zip(expected_values, calculated_values):
        for ve, vc in zip(expect, calc):
            np.testing.assert_allclose(ve, vc / np.log(2), rtol=0.04, atol=0.5)
    return


class TestMatchingEntropy(unittest.TestCase):
    def test_subsystem_count(self):
        self.assertEqual(14, len(get_all_2x2_non_corner((5, 7))))
        self.assertEqual(20, len(get_all_2x3_non_corner((5, 7))))
        self.assertEqual(3, len(get_all_3x3_non_corner((5, 7))))

    def test_2x2_entropy_pauli(self):
        expected_values = [(2., 1., 1.), (3., 3., 2.), (3.,)]

        backend_sim = Aer.get_backend('aer_simulator')
        x, y = 5, 7
        for qubits in get_all_2x2_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x2, expected_values, type='pauli')


if __name__ == '__main__':
    unittest.main()
