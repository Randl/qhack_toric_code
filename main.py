import numpy as np
from qiskit import Aer
from qiskit import IBMQ

from topo_braiding import em_braiding_phase
from topo_entropy import calculate_topo_entropy_pauli, calculate_topo_entropy_haar
from toric_code import get_toric_code
from toric_code_matching import get_plaquette_matching

tc = get_toric_code(9, 13)
print(tc.circ.num_qubits, tc.circ.depth())
print(tc.plaquette_x, tc.plaquette_y)

x, y = 5, 5
tc = get_toric_code(x, y)
print(tc.circ.num_qubits, tc.circ.depth())
print(tc.plaquette_x, tc.plaquette_y)
px, py = tc.plaquette_x, tc.plaquette_y

backend_sim = Aer.get_backend('qasm_simulator')


IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-sherbrooke', group='udes', project='quicophy')
backend_hardware = provider.get_backend('ibm_washington')
backend = backend_hardware

theta_em, err_theta_em = em_braiding_phase(backend, x, y)

print('theta_em = {:.3f} +/- {:.3f}'.format(theta_em, err_theta_em))

abc_2x2 = [(0, 1), (2,), (3,)]
for i in range(px):
    for j in range(py):
        if i in (0, px - 1) or j in (0, py - 1):
            continue
        plaq = get_plaquette_matching(j, i)
        print(plaq)
        topo_ent = calculate_topo_entropy_pauli(backend, (x, y), plaq, abc_2x2)
        print('topo entropy', topo_ent / np.log(2))
        topo_ent = calculate_topo_entropy_haar(backend, (x, y), plaq, abc_2x2)
        print('topo entropy', topo_ent / np.log(2))
