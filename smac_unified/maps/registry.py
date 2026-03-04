from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MapParams:
    n_agents: int
    n_enemies: int
    limit: int
    a_race: str
    b_race: str
    unit_type_bits: int
    map_type: str
    directory: str = 'SMAC_Maps'
    filename: str | None = None


MAP_PARAM_REGISTRY: Dict[str, MapParams] = {
    # Core SMAC maps.
    '3m': MapParams(3, 3, 60, 'T', 'T', 0, 'marines'),
    '8m': MapParams(8, 8, 120, 'T', 'T', 0, 'marines'),
    '25m': MapParams(25, 25, 150, 'T', 'T', 0, 'marines'),
    '5m_vs_6m': MapParams(5, 6, 70, 'T', 'T', 0, 'marines'),
    '8m_vs_9m': MapParams(8, 9, 120, 'T', 'T', 0, 'marines'),
    '10m_vs_11m': MapParams(10, 11, 150, 'T', 'T', 0, 'marines'),
    '27m_vs_30m': MapParams(27, 30, 180, 'T', 'T', 0, 'marines'),
    'MMM': MapParams(10, 10, 150, 'T', 'T', 3, 'MMM'),
    'MMM2': MapParams(10, 12, 180, 'T', 'T', 3, 'MMM'),
    '2s3z': MapParams(5, 5, 120, 'P', 'P', 2, 'stalkers_and_zealots'),
    '3s5z': MapParams(8, 8, 150, 'P', 'P', 2, 'stalkers_and_zealots'),
    '3s5z_vs_3s6z': MapParams(
        8, 9, 170, 'P', 'P', 2, 'stalkers_and_zealots'
    ),
    '3s_vs_3z': MapParams(3, 3, 150, 'P', 'P', 0, 'stalkers'),
    '3s_vs_4z': MapParams(3, 4, 200, 'P', 'P', 0, 'stalkers'),
    '3s_vs_5z': MapParams(3, 5, 250, 'P', 'P', 0, 'stalkers'),
    '1c3s5z': MapParams(
        9, 9, 180, 'P', 'P', 3, 'colossi_stalkers_zealots'
    ),
    '2m_vs_1z': MapParams(2, 1, 150, 'T', 'P', 0, 'marines'),
    'corridor': MapParams(6, 24, 400, 'P', 'Z', 0, 'zealots'),
    '6h_vs_8z': MapParams(6, 8, 150, 'Z', 'P', 0, 'hydralisks'),
    '2s_vs_1sc': MapParams(2, 1, 300, 'P', 'Z', 0, 'stalkers'),
    'so_many_baneling': MapParams(7, 32, 100, 'P', 'Z', 0, 'zealots'),
    'bane_vs_bane': MapParams(24, 24, 200, 'Z', 'Z', 2, 'bane'),
    '2c_vs_64zg': MapParams(2, 64, 400, 'P', 'Z', 0, 'colossus'),
    # SMACv2 generated maps.
    '10gen_terran': MapParams(
        10,
        10,
        200,
        'T',
        'T',
        3,
        'terran_gen',
        directory='SMAC_Maps',
        filename='32x32_flat',
    ),
    '10gen_zerg': MapParams(
        10,
        10,
        200,
        'Z',
        'Z',
        3,
        'zerg_gen',
        directory='SMAC_Maps',
        filename='32x32_flat',
    ),
    '10gen_protoss': MapParams(
        10,
        10,
        200,
        'P',
        'P',
        3,
        'protoss_gen',
        directory='SMAC_Maps',
        filename='32x32_flat',
    ),
    # SMAC-Hard extended map family (core-first set).
    '2vr_vs_3sc': MapParams(
        2, 3, 400, 'P', 'Z', 0, 'void_ray', directory='new_maps'
    ),
    '3hl_vs_24zl': MapParams(
        3, 24, 400, 'T', 'Z', 0, 'hellion', directory='new_maps'
    ),
    '3rp_vs_5zl': MapParams(
        3, 5, 400, 'T', 'P', 0, 'reaper', directory='new_maps'
    ),
    '3rp_vs_24zl': MapParams(
        3, 24, 400, 'T', 'Z', 0, 'reaper', directory='new_maps'
    ),
    '7q_vs_2bc': MapParams(
        7, 2, 400, 'Z', 'T', 0, 'qween', directory='new_maps'
    ),
    '3st_vs_5zl': MapParams(
        3, 5, 400, 'P', 'P', 0, 'stalkers', directory='new_maps'
    ),
    '6m_vs_10m': MapParams(
        6, 10, 400, 'T', 'T', 0, 'marines', directory='new_maps'
    ),
    'mmmt': MapParams(22, 22, 400, 'T', 'T', 4, 'MMM', directory='new_maps'),
    'mmmt_vs_zhb': MapParams(
        22, 46, 400, 'T', 'T', 4, 'MMM', directory='new_maps'
    ),
    'mmmt_vs_zspi': MapParams(
        22, 16, 400, 'T', 'P', 4, 'MMM', directory='new_maps'
    ),
}


_MAPS_REGISTERED = False


def register_maps() -> None:
    global _MAPS_REGISTERED
    if _MAPS_REGISTERED:
        return

    from pysc2.maps import lib

    class StandaloneSMACMap(lib.Map):
        directory = 'SMAC_Maps'
        download = 'https://github.com/oxwhirl/smac#smac-maps'
        players = 2
        step_mul = 8
        game_steps_per_episode = 0

    for map_name, params in MAP_PARAM_REGISTRY.items():
        attrs = {
            'directory': params.directory,
            'filename': params.filename or map_name,
            'players': 2,
            'step_mul': 8,
            'game_steps_per_episode': 0,
        }
        if map_name not in globals():
            globals()[map_name] = type(map_name, (StandaloneSMACMap,), attrs)
    _MAPS_REGISTERED = True


def resolve_map_directory(*, params: MapParams, opponent_mode: str) -> str:
    directory = str(params.directory or 'SMAC_Maps')
    if opponent_mode == 'scripted_pool' and directory == 'SMAC_Maps':
        return 'new_maps'
    return directory


def resolve_map_filename(*, map_name: str, params: MapParams) -> str:
    return str(params.filename or map_name)


def get_map_params(map_name: str) -> MapParams:
    params = MAP_PARAM_REGISTRY.get(map_name)
    if params is None:
        raise KeyError(
            f'Unknown SMAC map: {map_name!r}. '
            f'Known maps: {sorted(MAP_PARAM_REGISTRY)[:8]}...'
        )
    return params
