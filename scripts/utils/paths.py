#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
Path Resolution Module for INIA Project
=======================================

Central source of truth for all paths in the project. Supports:
- Environment variable overrides
- Config file settings
- Auto-detection of project root
- Cross-platform path handling

Usage:
    from scripts.utils.paths import get_paths, get_project_root, get_external_drive

    # Get all paths as a dict
    paths = get_paths()
    print(paths['external']['outputs_base'])

    # Get specific paths
    project_root = get_project_root()
    external_drive = get_external_drive()

Environment Variables:
    INIA_PROJECT_ROOT - Override project root detection
    INIA_EXTERNAL_DRIVE - Override external data location (default: <project_root>/data)

Configuration:
    configs/paths.yaml - Central path configuration
    .env - Local overrides (copy from .env.template)
"""

import os
import re
import sys
import platform
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Union
from functools import lru_cache

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'


def _find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Auto-detect project root by looking for marker files.

    Searches upward from start_path (or this file's location) until it finds
    a directory containing one of the marker files.

    Args:
        start_path: Starting directory for search (default: this file's dir)

    Returns:
        Path to project root

    Raises:
        RuntimeError: If project root cannot be found
    """
    markers = [
        'CITATION.cff',
        'environment.yml',
        'pyproject.toml',
        '.git',
    ]

    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    for _ in range(10):  # Max 10 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: assume standard location based on platform
    fallbacks = _get_platform_fallback_paths()
    for fallback in fallbacks:
        if fallback.exists():
            return fallback

    raise RuntimeError(
        f"Could not find project root. Set INIA_PROJECT_ROOT environment variable.\n"
        f"Searched from: {start_path}"
    )


def _get_platform_fallback_paths() -> list:
    """Get fallback project paths based on platform."""
    project_name = 'ragweed-multiscale-surveillance'
    return [
        Path.home() / 'dev' / project_name,
        Path.home() / project_name,
    ]


def _get_default_external_drive() -> Path:
    """
    Get default external data location.

    Returns:
        Path to the data directory (<project_root>/data). Set the
        INIA_EXTERNAL_DRIVE environment variable to point elsewhere.
    """
    return get_project_root() / 'data'


def _load_env_file(project_root: Path) -> None:
    """
    Load .env file if it exists and dotenv is available.

    Args:
        project_root: Project root directory
    """
    if not _HAS_DOTENV:
        return

    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)


def _substitute_variables(value: str, context: Dict[str, Any]) -> str:
    """
    Substitute ${variable} patterns in a string.

    Supports:
        ${var_name} - Simple variable reference
        ${section.key} - Nested dict reference

    Args:
        value: String with potential variable references
        context: Dict containing variable values

    Returns:
        String with variables substituted
    """
    if not isinstance(value, str):
        return value

    def replace_var(match):
        var_path = match.group(1)

        # Try environment variable first
        env_val = os.environ.get(var_path)
        if env_val is not None:
            return env_val

        # Try nested dict lookup
        parts = var_path.split('.')
        obj = context
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return match.group(0)  # Keep original if not found
        return str(obj) if obj is not None else match.group(0)

    pattern = r'\$\{([^}]+)\}'
    result = value
    for _ in range(5):  # Max 5 iterations for nested references
        new_result = re.sub(pattern, replace_var, result)
        if new_result == result:
            break
        result = new_result

    return result


def _resolve_paths(config: Dict[str, Any], base_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve all path variables in config.

    Args:
        config: Configuration dict with potential variable references
        base_context: Base context for variable substitution

    Returns:
        Config with all variables resolved
    """
    result = {}
    context = {**base_context, **config}

    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _resolve_paths(value, context)
        elif isinstance(value, str):
            result[key] = _substitute_variables(value, context)
        else:
            result[key] = value

    return result


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Get the project root directory.

    Priority:
        1. INIA_PROJECT_ROOT environment variable
        2. Auto-detection from file location

    Returns:
        Path to project root
    """
    env_root = os.environ.get('INIA_PROJECT_ROOT')
    if env_root:
        return Path(env_root)
    return _find_project_root()


@lru_cache(maxsize=1)
def get_external_drive() -> Path:
    """
    Get the external drive path.

    Priority:
        1. INIA_EXTERNAL_DRIVE environment variable
        2. Default: <project_root>/data

    Returns:
        Path to external drive
    """
    env_drive = os.environ.get('INIA_EXTERNAL_DRIVE')
    if env_drive:
        return Path(env_drive)
    return _get_default_external_drive()


def _load_config(project_root: Path) -> Dict[str, Any]:
    """
    Load paths configuration from YAML file.

    Args:
        project_root: Project root directory

    Returns:
        Configuration dict
    """
    config_path = project_root / 'configs' / 'paths.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, Any]:
    """
    Get all resolved paths for the project.

    Returns a dict with sections:
        - base: project_root, external_drive
        - external: paths on external drive
        - local: paths relative to project root
        - defaults: default filenames
        - shapefile_paths: image path resolution config

    All paths are resolved and ready to use.

    Returns:
        Dict with all resolved paths
    """
    project_root = get_project_root()
    external_drive = get_external_drive()

    # Load .env file
    _load_env_file(project_root)

    # Load config
    config = _load_config(project_root)

    # Build base context
    base_context = {
        'project_root': str(project_root),
        'external_drive': str(external_drive),
        'base': {
            'project_root': str(project_root),
            'external_drive': str(external_drive),
        }
    }

    # Resolve all paths
    resolved = _resolve_paths(config, base_context)

    # Ensure base paths are set
    if 'base' not in resolved:
        resolved['base'] = {}
    resolved['base']['project_root'] = str(project_root)
    resolved['base']['external_drive'] = str(external_drive)

    # Convert local paths to absolute
    if 'local' in resolved:
        for key, value in resolved['local'].items():
            if isinstance(value, str) and not value.startswith('/'):
                resolved['local'][key] = str(project_root / value)

    return resolved


def get_path(path_key: str, default: Optional[str] = None) -> Optional[Path]:
    """
    Get a specific path by key.

    Supports dot notation for nested keys: 'external.outputs_base'

    Args:
        path_key: Path key (e.g., 'external.outputs_base', 'local.configs')
        default: Default value if path not found

    Returns:
        Path object, or None if not found
    """
    paths = get_paths()
    parts = path_key.split('.')

    obj = paths
    for part in parts:
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return Path(default) if default else None

    if obj is None:
        return Path(default) if default else None

    return Path(obj)


def resolve_image_base(base_key: str) -> Optional[Path]:
    """
    Resolve an image base key to a path.

    Used for shapefile img_base field resolution.

    Args:
        base_key: Base key (e.g., 'EXTERNAL_OUTPUTS', 'LOCAL_OUTPUTS')

    Returns:
        Resolved path, or None if key not found
    """
    paths = get_paths()
    bases = paths.get('shapefile_paths', {}).get('bases', {})

    if base_key in bases:
        return Path(bases[base_key])

    # Fallback to direct path lookup
    return get_path(base_key.lower())


def resolve_image_path(img_base: Optional[str], img_path: str) -> Path:
    """
    Resolve full image path from base key and relative path.

    Args:
        img_base: Base key (e.g., 'EXTERNAL_OUTPUTS') or None for absolute
        img_path: Relative or absolute image path

    Returns:
        Absolute path to image
    """
    # If no base, treat img_path as absolute or project-relative
    if not img_base or img_base == '':
        path = Path(img_path)
        if path.is_absolute():
            return path
        return get_project_root() / img_path

    # Resolve base and combine
    base_path = resolve_image_base(img_base)
    if base_path:
        return base_path / img_path

    # Fallback: treat as absolute
    return Path(img_path)


def get_output_path(
    output_type: str,
    filename: Optional[str] = None,
    prefer_external: bool = True
) -> Path:
    """
    Get an output path based on type.

    Args:
        output_type: Type of output ('geo_exports', 'satellite_embeddings', etc.)
        filename: Optional filename to append
        prefer_external: If True, prefer external drive paths

    Returns:
        Path for the output
    """
    paths = get_paths()

    # Try external first if preferred
    if prefer_external and 'external' in paths:
        external_paths = paths['external']
        if output_type in external_paths:
            base = Path(external_paths[output_type])
            if base.parent.exists():  # External drive is mounted
                return base / filename if filename else base

    # Fall back to local
    if 'local' in paths:
        local_paths = paths['local']
        local_key = f'{output_type}_local' if f'{output_type}_local' in local_paths else output_type
        if local_key in local_paths:
            base = Path(local_paths[local_key])
            base.mkdir(parents=True, exist_ok=True)
            return base / filename if filename else base

    # Ultimate fallback
    fallback = get_project_root() / 'outputs' / output_type
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / filename if filename else fallback


def verify_paths(verbose: bool = True) -> Dict[str, bool]:
    """
    Verify that critical paths exist.

    Args:
        verbose: Print status messages

    Returns:
        Dict mapping path names to existence status
    """
    status = {}
    paths = get_paths()

    checks = [
        ('project_root', paths.get('base', {}).get('project_root')),
        ('external_drive', paths.get('base', {}).get('external_drive')),
        ('configs', paths.get('local', {}).get('configs')),
        ('external_outputs', paths.get('external', {}).get('outputs_base')),
        ('external_data', paths.get('external', {}).get('data_base')),
    ]

    if verbose:
        print("Path Verification")
        print("=" * 60)

    for name, path in checks:
        if path:
            exists = Path(path).exists()
            status[name] = exists
            if verbose:
                icon = '+' if exists else '-'
                print(f"  [{icon}] {name}: {path}")
        else:
            status[name] = False
            if verbose:
                print(f"  [?] {name}: NOT CONFIGURED")

    if verbose:
        print("=" * 60)
        valid = sum(status.values())
        total = len(status)
        print(f"  {valid}/{total} paths verified")

    return status


# Module-level convenience functions
def project_root() -> Path:
    """Alias for get_project_root()."""
    return get_project_root()


def external_drive() -> Path:
    """Alias for get_external_drive()."""
    return get_external_drive()


if __name__ == '__main__':
    # Run verification when executed directly
    import sys

    print("\nINIA Project Paths")
    print("=" * 60)
    print(f"Project Root: {get_project_root()}")
    print(f"External Drive: {get_external_drive()}")
    print()

    status = verify_paths(verbose=True)

    if not all(status.values()):
        print("\nSome paths are missing. Check configuration:")
        print("  - Set INIA_EXTERNAL_DRIVE if using different drive")
        print("  - Create .env file from .env.template")
        sys.exit(1)

    print("\nAll paths verified successfully.")
