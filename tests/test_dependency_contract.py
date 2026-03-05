from __future__ import annotations

import re
from pathlib import Path


def _project_block(pyproject_text: str) -> str:
    match = re.search(r'\[project\](.*?)(?:\n\[|$)', pyproject_text, flags=re.S)
    assert match is not None
    return str(match.group(1))


def _project_dependency_names(pyproject_text: str) -> set[str]:
    block = _project_block(pyproject_text)
    deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', block, flags=re.S)
    assert deps_match is not None
    raw_items = [
        item.strip().strip("'\"")
        for item in str(deps_match.group(1)).split(',')
        if item.strip()
    ]
    names: set[str] = set()
    for item in raw_items:
        name = re.split(r'[<>=!~ ]+', item, maxsplit=1)[0].strip()
        if name:
            names.add(name)
    return names


def test_pyproject_base_dependencies_include_native_runtime_requirements():
    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    text = pyproject.read_text(encoding='utf-8')
    dependency_names = _project_dependency_names(text)
    assert 'numpy' in dependency_names
    assert 'pysc2' in dependency_names
    assert 's2clientprotocol' in dependency_names


def test_pyproject_does_not_hide_native_runtime_in_optional_extra():
    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    text = pyproject.read_text(encoding='utf-8')
    assert 'native-backend' not in text
