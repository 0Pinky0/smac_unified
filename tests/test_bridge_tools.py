import sys

from tools.bridge_tools.legacy_bridge import _install_compat_shim


def test_bridge_tools_installs_compat_shim_for_legacy_wrappers():
    original = sys.modules.pop('smac_unified.compat', None)
    try:
        _install_compat_shim()
        compat_module = sys.modules.get('smac_unified.compat')
        assert compat_module is not None
        assert hasattr(compat_module, 'make_legacy_env')
        assert hasattr(compat_module, 'translate_legacy_kwargs')
        assert hasattr(compat_module, 'LegacyEnvAdapter')
    finally:
        if original is not None:
            sys.modules['smac_unified.compat'] = original
