import unittest

import numpy as np
from qiskit import transpile

from backends import get_clean_backend, get_noisy_backend, run_job, get_ibm_mock_backend
from toric_code import get_toric_code, calibrate_readout, fix_measurements
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


def get_plaquette_ev(backend, size, plaquette_index, shots=1024, run_kwargs=None, calibrate=False):
    x, y = size
    px, py = plaquette_index
    tc = get_toric_code(x, y)

    tc.measure_plaquette(py, px)
    _, counts = run_job(tc.circ, backend, shots=shots, run_kwargs=run_kwargs, calibrate=calibrate)
    return count_to_parity(counts)


def get_star_ev(backend, size, star_index, shots=1024, run_kwargs=None, calibrate=False):
    x, y = size
    sx, sy = star_index
    tc = get_toric_code(x, y)

    tc.measure_star(sx, sy)
    _, counts = run_job(tc.circ, backend, shots=shots, run_kwargs=run_kwargs, calibrate=calibrate)
    return count_to_parity(counts)


def run_plaquette_test(x, y, backend, run_kwargs, shots=1024, calibrate=False, rtol=1e-07):
    tc = get_toric_code(x, y)
    px, py = tc.plaquette_x, tc.plaquette_y
    for i in range(px):
        for j in range(py):
            ev = get_plaquette_ev(backend, (x, y), (i, j), shots, run_kwargs, calibrate)
            np.testing.assert_allclose(ev, 1, rtol=rtol)


def run_star_test(x, y, backend, run_kwargs, shots=1024, calibrate=False, rtol=1e-07):
    tc = get_toric_code(x, y)
    sx, sy = tc.star_x, tc.star_y

    for i in range(sx):
        for j in range(sy):
            if len(get_star_matching(i, j, tc.y, tc.x)) < 4:
                continue

            ev = get_star_ev(backend, (x, y), (i, j), shots, run_kwargs, calibrate)
            np.testing.assert_allclose(ev, 1, rtol=rtol)


class TestMatchingInit(unittest.TestCase):
    def test_plaquettes_clean(self):
        x, y = 5, 7
        backend_sim, run_kwargs = get_clean_backend()
        run_plaquette_test(x, y, backend_sim, run_kwargs, calibrate=False, rtol=1e-07)

    def test_plaquettes_noisy(self):
        x, y = 5, 7
        backend_sim, run_kwargs = get_noisy_backend(0.03)
        run_plaquette_test(x, y, backend_sim, run_kwargs, calibrate=True, rtol=0.05)

    def test_plaquettes_ibm(self):
        x, y = 5, 5
        backend_sim, run_kwargs = get_ibm_mock_backend('ibmq_mumbai')
        run_plaquette_test(x, y, backend_sim, run_kwargs, shots=10000, calibrate=True, rtol=0.3)

    def test_stars_clean(self):
        x, y = 5, 7
        backend_sim, run_kwargs = get_clean_backend()
        run_star_test(x, y, backend_sim, run_kwargs, calibrate=False, rtol=1e-07)

    def test_stars_noisy(self):
        x, y = 5, 5
        backend_sim, run_kwargs = get_noisy_backend(0.03)
        run_star_test(x, y, backend_sim, run_kwargs, calibrate=True, rtol=0.05)

    def test_stars_ibm(self):
        x, y = 5, 5
        backend_sim, run_kwargs = get_ibm_mock_backend('ibmq_mumbai')
        run_star_test(x, y, backend_sim, run_kwargs, calibrate=True, rtol=0.15)


if __name__ == '__main__':
    unittest.main()
