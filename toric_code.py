from toric_code_matching import ToricCodeMatching
from toric_code_mixed import ToricCodeMixed


def get_toric_code(x, y, classical_bit_count=4, ancillas_count=0, boundary_condition='matching'):
    assert boundary_condition in ('mixed', 'matching')
    if boundary_condition == 'matching':
        return ToricCodeMatching(x, y, classical_bit_count, ancillas_count)
    elif boundary_condition == 'mixed':
        return ToricCodeMixed(x, y, classical_bit_count, ancillas_count)
