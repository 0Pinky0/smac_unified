from smac_unified.handlers import CORE_PARITY_MATRIX, entries_for_block


def test_core_parity_matrix_covers_all_core_blocks():
    blocks = {
        'action',
        'observation_state',
        'reward_termination',
        'reset_init',
        'api_contract',
    }
    assert len(CORE_PARITY_MATRIX) >= len(blocks)
    for block in blocks:
        assert len(entries_for_block(block)) > 0


def test_core_parity_matrix_has_legacy_and_unified_symbols():
    for entry in CORE_PARITY_MATRIX:
        assert entry.legacy_symbol
        assert entry.unified_symbol
        assert '.' in entry.legacy_symbol
        assert '.' in entry.unified_symbol

