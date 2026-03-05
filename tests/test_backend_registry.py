from smac_unified import NormalizedEnvAdapter, SMACEnvCore, make_env


def test_make_env_constructs_native_core_env():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=False,
    )
    assert isinstance(env, SMACEnvCore)
    env.close()


def test_make_env_constructs_native_normalized_adapter():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=True,
    )
    assert isinstance(env, NormalizedEnvAdapter)
    env.close()


def test_make_env_hard_break_rejects_backend_mode_keyword():
    try:
        make_env(
            family='smac',
            map_name='3m',
            normalized_api=False,
            backend_mode='native',
        )
        raise AssertionError('Expected TypeError when passing backend_mode.')
    except TypeError as exc:
        assert 'backend_mode' in str(exc)
