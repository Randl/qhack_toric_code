import itertools
from collections import defaultdict

import numpy as np
from qiskit import transpile
from tqdm import tqdm, trange

from toric_code import get_toric_code
from toric_code_matching import is_inside_matching, is_corner_matching

ABC_DIVISION_2x2 = [(0, 1), (2,), (3,)]
ABC_DIVISION_2x3_RIGHT = [(1, 3), (0, 2), (4, 5)]
ABC_DIVISION_2x3_LEFT = [(2, 4), (0, 1), (3, 5)]
ABC_DIVISION_3x3 = [(0, 1, 3), (2, 5, 7), (4, 6, 8)]


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def purity_single_realization(counts):
    # counts is dictionary {s: count}
    N = sum(counts.values())
    q_cnt = len(list(counts.keys())[0])
    pur = 0
    for s1 in counts:
        p1 = counts[s1] / N
        for s2 in counts:
            factor = (-2) ** (-hamming_distance(s1, s2))
            p2 = counts[s2] / N
            pur += 2 ** q_cnt * factor * p1 * (N * p2 - 1) / (N - 1)
    return pur


def second_renyi_entropy(all_counts):
    # all_counts is list of dictionaries {s: count}
    return -np.log(sum([purity_single_realization(counts) for counts in tqdm(all_counts)]) / len(all_counts))


def get_subsystem_counts(full_counts, sub_idx):
    all_subsystem_counts = []
    for counts in full_counts:
        subsystem_counts = defaultdict(lambda: 0)
        for s in counts:
            s_sub = ''.join(s[i] for i in sub_idx)
            subsystem_counts[s_sub] += counts[s]
        all_subsystem_counts.append(subsystem_counts)
    return all_subsystem_counts


def subsystem_sre(full_counts, sub_idx):
    return second_renyi_entropy(get_subsystem_counts(full_counts, sub_idx))


def calculate_s_subsystems(full_counts, subsystems):
    one = [subsystem_sre(full_counts, sub_idx) for sub_idx in subsystems]
    two_subsystems = [subsystems[0] + subsystems[1], subsystems[0] + subsystems[2], subsystems[1] + subsystems[2]]
    two = [subsystem_sre(full_counts, sub_idx) for sub_idx in two_subsystems]
    three = [second_renyi_entropy(full_counts)]
    return one, two, three


def calculate_s_topo(full_counts, subsystems):
    one, two, three = calculate_s_subsystems(full_counts, subsystems)
    print([o / np.log(2) for o in one], [t / np.log(2) for t in two], [t / np.log(2) for t in three])
    return sum(one) - sum(two) + sum(three)


def calculate_topo_entropy_pauli(backend, size, qubits, subsystems):
    x, y = size
    all_counts = []
    all_gates = [''.join(x) for x in itertools.product('xyz', repeat=len(qubits))]
    for gates in tqdm(all_gates):
        tc = get_toric_code(x, y, len(qubits))
        tc.measure_pauli(qubits, gates)

        job = backend.run(transpile(tc.circ, backend), shots=1024)
        result = job.result()
        counts = result.get_counts(tc.circ)
        all_counts.append(counts)
    return calculate_s_topo(all_counts, subsystems)


def calculate_topo_entropy_haar(backend, size, qubits, subsystems):
    x, y = size
    all_counts = []
    for _ in trange(100):
        tc = get_toric_code(x, y, len(qubits))
        tc.measure_haar(qubits)

        job = backend.run(transpile(tc.circ, backend), shots=1024)
        result = job.result()
        counts = result.get_counts(tc.circ)
        all_counts.append(counts)
    return calculate_s_topo(all_counts, subsystems)


def get_all_2x2_non_corner(size):
    x, y = size
    all_sys = []
    for rx in range(x):
        for ry in range(y):
            if rx % 2 == 0:
                sys = (rx, ry), (rx + 1, ry), (rx + 1, ry + 1), (rx + 2, ry)
            else:
                sys = (rx, ry), (rx + 1, ry - 1), (rx + 1, ry), (rx + 2, ry)
            if all([is_inside_matching(size, s) for s in sys]) and not any([is_corner_matching(size, s) for s in sys]):
                all_sys.append(sys)
    return all_sys


def get_all_2x3_left_non_corner(size):
    x, y = size
    all_sys = []
    for rx in range(x):
        for ry in range(y):
            if rx % 2 == 0:
                sys_l = (rx, ry), (rx + 1, ry), (rx + 1, ry + 1), (rx + 2, ry), (rx + 2, ry + 1), (rx + 3, ry + 1)
            else:
                sys_l = (rx, ry), (rx + 1, ry - 1), (rx + 1, ry), (rx + 2, ry), (rx + 2, ry + 1), (rx + 3, ry)
            if all([is_inside_matching(size, s) for s in sys_l]) and not any(
                    [is_corner_matching(size, s) for s in sys_l]):
                all_sys.append(sys_l)
    return all_sys


def get_all_2x3_right_non_corner(size):
    x, y = size
    all_sys = []
    for rx in range(x):
        for ry in range(y):
            if rx % 2 == 0:
                sys_r = (rx, ry), (rx + 1, ry), (rx + 1, ry + 1), (rx + 2, ry - 1), (rx + 2, ry), (rx + 3, ry)
            else:
                sys_r = (rx, ry), (rx + 1, ry - 1), (rx + 1, ry), (rx + 2, ry - 1), (rx + 2, ry), (rx + 3, ry - 1)
            if all([is_inside_matching(size, s) for s in sys_r]) and not any(
                    [is_corner_matching(size, s) for s in sys_r]):
                all_sys.append(sys_r)
    return all_sys


def get_all_2x3_non_corner(size):
    return get_all_2x3_left_non_corner(size) + get_all_2x3_right_non_corner(size)


def get_all_3x3_non_corner(size):
    x, y = size
    all_sys = []
    for rx in range(x):
        for ry in range(y):
            if rx % 2 == 0:
                sys = [(-1, -1)]  # skip, why
            else:
                sys = (rx, ry), (rx + 1, ry - 1), (rx + 1, ry), (rx + 2, ry - 1), (rx + 2, ry), (rx + 2, ry + 1), (rx + 3, ry - 1), (rx + 3, ry), (rx + 4, ry),
            if all([is_inside_matching(size, s) for s in sys]) and not any([is_corner_matching(size, s) for s in sys]):
                all_sys.append(sys)
    return all_sys
