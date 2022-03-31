from qiskit import Aer, IBMQ

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error


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


def get_noisy_backend(p):
    return Aer.get_backend('aer_simulator'), {'noise_model': get_noise(p)}
