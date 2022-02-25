import unittest

import numpy as np
from qiskit import Aer
from qiskit import transpile

from topo_braiding import create_e_particles, create_m_particles, apply_cxxxx_on_square, apply_cxxyyzz_on_rectangle
from toric_code import get_toric_code


class TestBraiding(unittest.TestCase):
    def test_em_braiding(self):
        x, y = 5, 7
        backend = Aer.get_backend('aer_simulator')
        for sq in [(0, 2), (0, 3), (2, 2), (2, 1)]:
            tc = get_toric_code(x, y, classical_bit_count=1, ancillas_count=1)

            x_string = [(2, 1), (2, 2)]
            create_e_particles(tc, x_string)

            z_string = [(1, 4), (0, 3), (1, 3)]  # , (0,2), (1,2)]
            create_m_particles(tc, z_string)

            tc.circ.h(tc.ancillas[0])
            apply_cxxxx_on_square(tc, sq)
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
            print(counts)

            cos_theta = (counts['0'] - counts['1']) / Nshots
            expected_res = -1. if sq in [(0, 2), (0, 3)] else 1.
            np.testing.assert_allclose(cos_theta, expected_res)

    def test_psipsi_exchange(self):
        x, y = 5, 7
        backend = Aer.get_backend('aer_simulator')
        sq = (0, 3)
        tc = get_toric_code(x, y, classical_bit_count=1, ancillas_count=1)

        x_strings = [[(2, 1), (2, 2)],
                     [(1, 4), (3, 4)]]
        for x_string in x_strings:
            create_e_particles(tc, x_string)

        z_strings = [[(2, 3), (3, 3), (4, 3)],
                     [(3, 1), (4, 1), (3, 2)], ]
        for z_string in z_strings:
            create_m_particles(tc, z_string)

        tc.circ.h(tc.ancillas[0])
        apply_cxxyyzz_on_rectangle(tc, sq)
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
        print(counts)

        cos_theta = (counts['0'] - counts['1']) / Nshots
        expected_res = -1.
        np.testing.assert_allclose(cos_theta, expected_res)

    def test_psione_braiding(self):
        x, y = 5, 7
        backend = Aer.get_backend('aer_simulator')
        sq = (0, 3)
        tc = get_toric_code(x, y, classical_bit_count=1, ancillas_count=1)

        x_strings = [[(1, 4), (3, 4)]]
        for x_string in x_strings:
            create_e_particles(tc, x_string)

        z_strings = [[(2, 3), (3, 3), (4, 3)], ]
        for z_string in z_strings:
            create_m_particles(tc, z_string)

        tc.circ.h(tc.ancillas[0])
        apply_cxxyyzz_on_rectangle(tc, sq)
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
        print(counts)

        cos_theta = (counts['0'] - counts['1']) / Nshots
        expected_res = 1.
        np.testing.assert_allclose(cos_theta, expected_res)

    def test_e_trivial_braiding(self):
        x, y = 5, 7
        backend = Aer.get_backend('aer_simulator')
        for sq in [(0, 2), (0, 3), (2, 2), (2, 1)]:
            tc = get_toric_code(x, y, classical_bit_count=1, ancillas_count=1)

            x_string = [(2, 1), (2, 2)]
            create_e_particles(tc, x_string)

            tc.circ.h(tc.ancillas[0])
            apply_cxxxx_on_square(tc, sq)
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
            print(counts)

            cos_theta = (counts['0'] - counts['1']) / Nshots
            expected_res = 1.
            np.testing.assert_allclose(cos_theta, expected_res)


if __name__ == '__main__':
    unittest.main()
