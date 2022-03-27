import unittest

import numpy as np
from qiskit import transpile

from backends import get_clean_backend, get_noisy_backend
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


def get_plaquette_ev(backend, size, plaquette_index, run_kwargs=None, calibrate=False):
    x, y = size
    px, py = plaquette_index
    tc = get_toric_code(x, y)

    tc.measure_plaquette(py, px)
    job = backend.run(transpile(tc.circ, backend), shots=1024, **run_kwargs)
    result = job.result()
    if calibrate:
        meas_fitter = calibrate_readout(tc.measured_qubits, backend, run_kwargs)
        counts = fix_measurements(meas_fitter, result)
        # print(count_to_parity(result.get_counts(tc.circ)), count_to_parity(counts))
    else:
        counts = result.get_counts(tc.circ)
    return count_to_parity(counts)


def get_star_ev(backend, size, star_index, run_kwargs=None, calibrate=False):
    x, y = size
    sx, sy = star_index
    tc = get_toric_code(x, y)

    tc.measure_star(sx, sy)
    job = backend.run(transpile(tc.circ, backend), shots=1024, **run_kwargs)
    result = job.result()
    if calibrate:
        meas_fitter = calibrate_readout(tc.measured_qubits, backend, run_kwargs)
        counts = fix_measurements(meas_fitter, result)
        # print(count_to_parity(result.get_counts(tc.circ)), count_to_parity(counts))
    else:
        counts = result.get_counts(tc.circ)
    return count_to_parity(counts)


class TestMatchingInit(unittest.TestCase):
    def test_plaquettes(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        px, py = tc.plaquette_x, tc.plaquette_y

        # Use Aer's simulator
        backend_sim, noise_model, coupling_map, basis_gates = get_clean_backend()
        run_kwargs = {'noise_model': noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates}
        for i in range(px):
            for j in range(py):
                ev = get_plaquette_ev(backend_sim, (x, y), (i, j), run_kwargs, False)
                np.testing.assert_allclose(ev, 1)

    def test_plaquettes_noisy(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        px, py = tc.plaquette_x, tc.plaquette_y

        # Use Aer's simulator
        backend_sim, noise_model, coupling_map, basis_gates = get_noisy_backend(0.03)
        run_kwargs = {'noise_model': noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates}
        for i in range(px):
            for j in range(py):
                ev = get_plaquette_ev(backend_sim, (x, y), (i, j), run_kwargs, True)
                np.testing.assert_allclose(ev, 1, atol=0.05)

    def test_stars(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        sx, sy = tc.star_x, tc.star_y

        # Use Aer's simulator
        backend_sim, noise_model, coupling_map, basis_gates = get_clean_backend()
        run_kwargs = {'noise_model': noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates}

        for i in range(sx):
            for j in range(sy):
                if len(get_star_matching(i, j, tc.y, tc.x)) < 4:
                    continue

                ev = get_star_ev(backend_sim, (x, y), (i, j), run_kwargs)
                np.testing.assert_allclose(ev, 1)

    def test_stars_noisy(self):
        x, y = 5, 7
        tc = get_toric_code(x, y)
        sx, sy = tc.star_x, tc.star_y

        # Use Aer's simulator
        backend_sim, noise_model, coupling_map, basis_gates = get_noisy_backend(0.03)
        run_kwargs = {'noise_model': noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates}

        for i in range(sx):
            for j in range(sy):
                if len(get_star_matching(i, j, tc.y, tc.x)) < 4:
                    continue

                ev = get_star_ev(backend_sim, (x, y), (i, j), run_kwargs, True)
                np.testing.assert_allclose(ev, 1, atol=0.05)


if __name__ == '__main__':
    unittest.main()
