import unittest

import numpy as np
from qiskit import Aer
from qiskit import transpile

from toric_code import get_toric_code
from toric_code_matching import get_star_matching


def count_to_parity(counts):
    one, minus_one = 0, 0
    for c in counts:
        if c.count('1') % 2 == 1:
            minus_one += counts[c]
        else:
            one += counts[c]
    parity = (one - minus_one) / (one + minus_one)
    return parity


def get_plaquette_ev(backend, size, plaquette_index):
    x, y = size
    px, py = plaquette_index
    tc = get_toric_code(x, y)

    tc.measure_plaquette(py, px)
    job = backend.run(transpile(tc.circ, backend), shots=1024)
    result = job.result()
    counts = result.get_counts(tc.circ)
    return count_to_parity(counts)


def get_star_ev(backend, size, star_index):
    x, y = size
    sx, sy = star_index
    tc = get_toric_code(x, y)

    tc.measure_star(sx, sy)
    job = backend.run(transpile(tc.circ, backend), shots=1024)
    result = job.result()
    counts = result.get_counts(tc.circ)
    return count_to_parity(counts)


class TestMatchingInit(unittest.TestCase):
    def test_plaquettes(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        px, py = tc.plaquette_x, tc.plaquette_y

        # Use Aer's simulator
        backend_sim = Aer.get_backend('aer_simulator')
        for i in range(px):
            for j in range(py):
                ev = get_plaquette_ev(backend_sim, (x, y), (i, j))
                np.testing.assert_allclose(ev, 1)

    def test_stars(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        sx, sy = tc.star_x, tc.star_y

        # Use Aer's simulator
        backend_sim = Aer.get_backend('aer_simulator')

        for i in range(sx):
            for j in range(sy):
                if len(get_star_matching(i, j, tc.y, tc.x)) < 4:
                    continue

                ev = get_star_ev(backend_sim, (x, y), (i, j))
                np.testing.assert_allclose(ev, 1)


if __name__ == '__main__':
    unittest.main()
