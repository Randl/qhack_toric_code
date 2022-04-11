from collections.abc import Iterable

from qiskit import Aer, IBMQ
from qiskit import transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.test.mock import FakeArmonk, FakeBelem, FakeBogota, FakeBrooklyn, FakeGuadalupe, FakeJakarta, FakeLagos, \
    FakeLima, FakeManila, FakeMontreal, FakeMumbai, FakeQuito, FakeSantiago, FakeToronto

from toric_code import calibrate_readout, fix_measurements


def get_noise(p):
    error_meas = pauli_error([('X', p), ('I', 1 - p)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements

    return noise_model


def get_clean_backend():
    return Aer.get_backend('aer_simulator'), {}


def get_ibm_sim_backend(backend_name='ibmq_manila', hub=None, group=None, project=None):
    # Build noise model from backend properties
    if hub is not None:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    else:
        provider = IBMQ.load_account()
    backend = provider.get_backend(backend_name)
    noise_model = NoiseModel.from_backend(backend)

    # Get coupling map from backend
    coupling_map = backend.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    return Aer.get_backend('aer_simulator'), {'noise_model': noise_model, 'coupling_map': coupling_map,
                                              'basis_gates': basis_gates}


def get_ibm_backend(backend_name='ibmq_manila', hub=None, group=None, project=None):
    # Build noise model from backend properties
    if hub is not None:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    else:
        provider = IBMQ.load_account()
    backend = provider.get_backend(backend_name)

    return backend, {}


name_to_backend = {'ibmq_armonk': FakeArmonk,
                   'ibmq_belem': FakeBelem,
                   'ibmq_bogota': FakeBogota,
                   'ibmq_brooklyn': FakeBrooklyn,
                   'ibmq_guadalupe': FakeGuadalupe,
                   'ibmq_jakarta': FakeJakarta,
                   'ibmq_lagos': FakeLagos,
                   'ibmq_lima': FakeLima,
                   'ibmq_manila': FakeManila,
                   'ibmq_montreal': FakeMontreal,
                   'ibmq_mumbai': FakeMumbai,
                   'ibmq_quito': FakeQuito,
                   'ibmq_santiago': FakeSantiago,
                   'ibmq_toronto': FakeToronto}


def get_ibm_mock_noise_backend(backend_name='ibmq_mumbai'):
    device = name_to_backend[backend_name]()
    noise_model = NoiseModel.from_backend(device)
    conf = device.configuration()
    kwargs = {
        "noise_model": noise_model,
        "coupling_map": conf.coupling_map,
        "basis_gates": conf.basis_gates,
    }
    return Aer.get_backend('aer_simulator'), kwargs


def get_ibm_mock_backend(backend_name='ibmq_mumbai'):
    device = name_to_backend[backend_name]()
    return device, {}


def get_noisy_backend(p):
    return Aer.get_backend('aer_simulator'), {'noise_model': get_noise(p)}


def get_best_transpilation(circ, backend, opt=None, transpile_attempts=10):
    our_circ = circ if isinstance(circ, Iterable) else [circ]
    transpiled = transpile(our_circ, backend, optimization_level=opt)
    for i in range(transpile_attempts - 1):
        tmp = transpile(our_circ, backend, optimization_level=opt)
        for j in range(len(transpiled)):
            if tmp[j].depth() < transpiled[j].depth():
                transpiled[j] = tmp[j]
    return transpiled if isinstance(circ, Iterable) else transpiled[0]


def get_measured_qubit_list(c):
    measure_ops = [x for x in c.data if x[0].name == 'measure']
    m_dict = {}
    for op in measure_ops:
        i, qubits, clbits = op
        for cb, qb in zip(clbits, qubits):
            m_dict[cb.index] = qb.index
    qubit_list = [0 for _ in range(max(m_dict.keys()) + 1)]
    for i in m_dict:
        qubit_list[i] = m_dict[i]
    return qubit_list


def run_job(circ, backend, shots, run_kwargs=None, calibrate=False, measured_qubits=None, opt=3,
            transpile_attempts=30):
    # TODO: multiple circuits in single job
    if run_kwargs is None:
        run_kwargs = {}
    transpiled_circ = get_best_transpilation(circ, backend, opt=opt, transpile_attempts=transpile_attempts)

    job = backend.run(transpiled_circ, shots=shots, **run_kwargs)
    result = job.result()
    if calibrate:
        qubit_list = get_measured_qubit_list(transpiled_circ)
        meas_fitter = calibrate_readout(qubit_list, backend, run_kwargs)
        counts = fix_measurements(meas_fitter, result)
    else:
        counts = result.get_counts(circ)
    return result, counts
