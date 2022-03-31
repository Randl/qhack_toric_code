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


def get_ibm_mock_backend(backend_name='ibmq_mumbai'):
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
    device = name_to_backend[backend_name]()
    noise_model = NoiseModel.from_backend(device)
    conf = device.configuration()
    kwargs = {
        "noise_model": noise_model,
        "coupling_map": conf.coupling_map,
        "basis_gates": conf.basis_gates,
    }
    return Aer.get_backend('aer_simulator'), kwargs


def get_noisy_backend(p):
    return Aer.get_backend('aer_simulator'), {'noise_model': get_noise(p)}


def run_job(circ, backend, shots, run_kwargs=None, calibrate=False, measured_qubits=None):
    #TODO: multiple circuits in single job
    if run_kwargs is None:
        run_kwargs={}
    job = backend.run(transpile(circ, backend), shots=shots, **run_kwargs)
    result = job.result()
    if calibrate:
        meas_fitter = calibrate_readout(measured_qubits, backend, run_kwargs)
        counts = fix_measurements(meas_fitter, result)
    else:
        counts = result.get_counts(circ)
    return result, counts
