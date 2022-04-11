from qiskit import transpile, QuantumRegister
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter

from toric_code_matching import ToricCodeMatching
from toric_code_mixed import ToricCodeMixed


def get_toric_code(x, y, classical_bit_count=4, ancillas_count=0, boundary_condition='matching'):
    assert boundary_condition in ('mixed', 'matching')
    if boundary_condition == 'matching':
        return ToricCodeMatching(x, y, classical_bit_count, ancillas_count)
    elif boundary_condition == 'mixed':
        return ToricCodeMixed(x, y, classical_bit_count, ancillas_count)


def calibrate_readout(qubit_indices, backend, run_kwargs=None):
    reg = QuantumRegister(len(qubit_indices))
    meas_calibs, state_labels = complete_meas_cal(qr=reg, circlabel='mcal')
    # Execute the calibration circuits without noise
    t_qc = transpile(meas_calibs, backend, initial_layout=qubit_indices, optimization_level=0)
    cal_results = backend.run(t_qc, shots=10000, **run_kwargs).result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    return meas_fitter


def fix_measurements(meas_fitter, results):
    # Get the filter object
    meas_filter = meas_fitter.filter

    # Results with mitigation
    mitigated_results = meas_filter.apply(results)
    mitigated_counts = mitigated_results.get_counts()
    return mitigated_counts
