from qiskit import Aer, IBMQ

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error


def get_noise(p):
    error_meas = pauli_error([('X', p), ('I', 1 - p)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements

    return noise_model


def get_clean_backend():
    return Aer.get_backend('aer_simulator'), None, None, None


def get_ibm_sim_backend():
    # Build noise model from backend properties
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_vigo')
    noise_model = NoiseModel.from_backend(backend)

    # Get coupling map from backend
    coupling_map = backend.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    return Aer.get_backend('aer_simulator'), noise_model, coupling_map, basis_gates


def get_noisy_backend(p):
    return Aer.get_backend('aer_simulator'), get_noise(p), None, None
