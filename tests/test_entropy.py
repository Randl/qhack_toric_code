import itertools
import unittest

import numpy as np
from qiskit import Aer
from qiskit import transpile
from tqdm import tqdm

from topo_entropy import ABC_DIVISION_2x2, ABC_DIVISION_2x3_LEFT, ABC_DIVISION_2x3_RIGHT, ABC_DIVISION_3x3
from topo_entropy import calculate_s_subsystems
from topo_entropy import get_all_2x2_non_corner, get_all_2x3_non_corner, get_all_3x3_non_corner
from topo_entropy import get_all_2x3_left_non_corner, get_all_2x3_right_non_corner
from toric_code import get_toric_code


def test_topo_entropy(backend, size, qubits, subsystems, expected_values, type='haar', cnt=1000, rtol=0.05):
    assert type in ('haar', 'pauli')
    print(qubits)
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

        job = backend.run(transpile(tc.circ, backend), shots=15000)  # note: number of shots is important
        result = job.result()
        counts = result.get_counts(tc.circ)
        all_counts.append(counts)
    calculated_values = calculate_s_subsystems(all_counts, subsystems)
    print(expected_values, [c / np.log(2) for calc in calculated_values for c in calc])
    for expect, calc in zip(expected_values, calculated_values):
        for ve, vc in zip(expect, calc):
            np.testing.assert_allclose(vc / np.log(2), ve, rtol=rtol, atol=0.)
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
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x2, expected_values, type='pauli', rtol=0.01)

    def test_2x2_entropy_haar(self):
        expected_values = [(2., 1., 1.), (3., 3., 2.), (3.,)]

        backend_sim = Aer.get_backend('aer_simulator')
        x, y = 5, 7
        for qubits in get_all_2x2_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x2, expected_values, type='haar', cnt=100,
                              rtol=0.05)

    def test_2x3_entropy_pauli(self):
        expected_values = [(2., 2., 2.), (4., 3., 4.), (4.,)]

        backend_sim = Aer.get_backend('aer_simulator')
        x, y = 5, 7
        for qubits in get_all_2x3_left_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x3_LEFT, expected_values, type='pauli',
                              rtol=0.03)
        for qubits in get_all_2x3_right_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x3_RIGHT, expected_values, type='pauli',
                              rtol=0.03)

    def test_2x3_entropy_haar(self):
        expected_values = [(2., 2., 2.), (4., 4., 3.), (4.,)]

        backend_sim = Aer.get_backend('aer_simulator')
        x, y = 5, 7
        for qubits in get_all_2x3_right_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x3_RIGHT, expected_values, type='haar',
                              cnt=100)
        for qubits in get_all_2x3_left_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_2x3_LEFT, expected_values, type='haar', cnt=100)

    def test_3x3_entropy_haar(self):
        expected_values = [(3., 3., 3.), (6., 5., 4.), (5.,)]

        backend_sim = Aer.get_backend('aer_simulator')
        x, y = 5, 7
        for qubits in get_all_3x3_non_corner((x, y)):
            test_topo_entropy(backend_sim, (x, y), qubits, ABC_DIVISION_3x3, expected_values, type='haar', cnt=1000)


if __name__ == '__main__':
    unittest.main()
