from tqdm import tqdm

import itertools

import numpy as np
from qiskit import transpile
from tqdm import tqdm

from toric_code import ToricCode


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
            pur += factor * p1 * p2  # TODO unbias?
    # print(q_cnt, np.log(pur)/np.log(2), -q_cnt-np.log(pur)/np.log(2))
    # -log(pur) = -qcnt* log(2) - log(por)
    return 2 ** q_cnt * pur


def second_renyi_entropy(all_counts):
    # all_counts is list of dictionaries {s: count}
    return -np.log(sum([purity_single_realization(counts) for counts in all_counts]) / len(all_counts))


def get_subsystem_counts(full_counts, sub_idx):
    all_subsystem_counts = []
    for counts in full_counts:
        subsystem_counts = {}
        for s in counts:
            s_sub = ''.join(s[i] for i in sub_idx)
            subsystem_counts[s_sub] = counts[s]
        all_subsystem_counts.append(subsystem_counts)
    return all_subsystem_counts


def subsystem_sre(full_counts, sub_idx):
    return second_renyi_entropy(get_subsystem_counts(full_counts, sub_idx))


def calculate_s_topo(full_counts, subsystems):
    one = [subsystem_sre(full_counts, sub_idx) for sub_idx in subsystems]
    two_subsystems = [subsystems[0] + subsystems[1], subsystems[0] + subsystems[2], subsystems[1] + subsystems[2]]
    two = [subsystem_sre(full_counts, sub_idx) for sub_idx in two_subsystems]
    three = [second_renyi_entropy(full_counts)]
    print([o / np.log(2) for o in one], [t / np.log(2) for t in two], [t / np.log(2) for t in three])
    return sum(one) - sum(two) + sum(three)


def calculate_topo_entropy_pauli(backend, qubits, subsystems):
    all_counts = []
    all_gates = [''.join(x) for x in itertools.product('xyz', repeat=len(qubits))]
    for gates in tqdm(all_gates):
        tc = ToricCode(3, 5, len(qubits))
        tc.measure_pauli(qubits, gates)

        job = backend.run(transpile(tc.circ, backend), shots=1024)
        result = job.result()
        counts = result.get_counts(tc.circ)
        all_counts.append(counts)
    return calculate_s_topo(all_counts, subsystems)
