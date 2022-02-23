import numpy as np
from qiskit import Aer
from qiskit import transpile

from topo_entropy import calculate_topo_entropy_pauli
from toric_code import ToricCode, get_star

tc = ToricCode(9, 13)
print(tc.circ.num_qubits, tc.circ.depth())
print(tc.plaquette_x, tc.plaquette_y)
tc = ToricCode(3, 5)
px, py = tc.plaquette_x, tc.plaquette_y
for i in range(px):
    for j in range(py):

        tc = ToricCode(3, 5)
        tc.measure_plaquette(i, j)

        # Use Aer's qasm_simulator
        backend_sim = Aer.get_backend('qasm_simulator')

        # Execute the circuit on the qasm simulator.
        # We've set the number of repeats of the circuit
        # to be 1024, which is the default.
        job_sim = backend_sim.run(transpile(tc.circ, backend_sim), shots=1024)

        # Grab the results from the job.
        result_sim = job_sim.result()
        counts = result_sim.get_counts(tc.circ)
        one, minus_one = 0, 0
        for c in counts:
            if c.count('1') % 2 == 1:
                minus_one += counts[c]
            else:
                one += counts[c]
        print('plaquett', i, j, (one - minus_one) / (one + minus_one))

sx, sy = tc.star_x, tc.star_y
for i in range(sx):
    for j in range(sy):
        if len(get_star(i, j, tc.y, tc.x)) < 4:
            continue
        tc = ToricCode(3, 5)
        tc.measure_star(i, j)

        # Use Aer's qasm_simulator
        backend_sim = Aer.get_backend('qasm_simulator')

        # Execute the circuit on the qasm simulator.
        # We've set the number of repeats of the circuit
        # to be 1024, which is the default.
        job_sim = backend_sim.run(transpile(tc.circ, backend_sim), shots=1024)

        # Grab the results from the job.
        result_sim = job_sim.result()
        counts = result_sim.get_counts(tc.circ)
        one, minus_one = 0, 0
        for c in counts:
            if c.count('1') % 2 == 1:
                minus_one += counts[c]
            else:
                one += counts[c]
        print('star', i, j, (one - minus_one) / (one + minus_one))

backend_sim = Aer.get_backend('qasm_simulator')
topo_ent = calculate_topo_entropy_pauli(backend_sim, [(1, 1), (2, 0), (2, 1), (3, 1)], [(0, 1), (2,), (3,)])
print('topo entropy', topo_ent / np.log(2))
