from smac_unified.backends import BackendConfig, BackendRegistry


class _DummyNative:
    family = "smac"
    kind = "native"
    priority = 1

    def is_available(self, config: BackendConfig):
        return True

    def make_env(self, config: BackendConfig):
        return config


class _DummyBridge:
    family = "smac"
    kind = "bridge"
    priority = 99

    def is_available(self, config: BackendConfig):
        return True

    def make_env(self, config: BackendConfig):
        return config


def test_registry_prefers_native_in_auto_mode():
    registry = BackendRegistry()
    registry.register(_DummyBridge())
    registry.register(_DummyNative())
    resolved = registry.get("smac", mode="auto", config=BackendConfig(family="smac"))
    assert resolved.kind == "native"


def test_registry_can_force_bridge_mode():
    registry = BackendRegistry()
    registry.register(_DummyNative())
    registry.register(_DummyBridge())
    resolved = registry.get("smac", mode="bridge", config=BackendConfig(family="smac"))
    assert resolved.kind == "bridge"
