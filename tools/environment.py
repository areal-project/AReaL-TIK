"""Record the actual build/runtime dependencies without installing anything."""

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {'command': command, 'exit_code': result.returncode,
                'output': (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'exit_code': None, 'error': str(error)}


def cutlass_identity(include):
    if not include:
        return None
    include = Path(include).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(p for folder in ('cute', 'cutlass')
                   for p in (include / folder).rglob('*') if p.is_file())
    if not files:
        raise FileNotFoundError(f'No CUTLASS/CuTe headers in {include}')
    for path in files:
        digest.update(path.relative_to(include).as_posix().encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    version_file = include / 'cutlass/version.h'
    text = version_file.read_text() if version_file.is_file() else ''
    parts = [re.search(r'^#define CUTLASS_' + field + r'\s+(\d+)', text, re.M)
             for field in ('MAJOR', 'MINOR', 'PATCH')]
    version = '.'.join(part.group(1) for part in parts) if all(parts) else None
    return {'include_directory': str(include), 'header_version': version,
            'header_tree_sha256': digest.hexdigest(), 'header_files': len(files),
            'hash_method': 'sorted relative path + NUL + SHA256(file bytes), for cute/ and cutlass/'}


def module_identity(name):
    try:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            return None
        path = Path(spec.origin)
        return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    except (ImportError, ValueError):
        return None


def collect_environment(config):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get('Name')}
    nvcc = str(Path(CUDA_HOME) / 'bin/nvcc') if CUDA_HOME else shutil.which('nvcc') or 'nvcc'
    cxx = shlex.split(os.environ.get('CXX', 'c++'))
    include_env = config['build'].get('cutlass_include_env')
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': properties.name, 'device_index': device,
        'device_uuid': str(getattr(properties, 'uuid', 'unavailable')),
        'capability': list(torch.cuda.get_device_capability(device)),
        'device_memory_bytes': properties.total_memory,
        'packages': dict(sorted(packages.items())),
        'baseline_modules': {name: module_identity(name) for name in
                             ('flash_attn', 'flash_attn_interface', 'flash_attn_3_cuda', 'flashinfer')},
        'nvcc': command_output([nvcc, '--version']),
        'cxx': command_output(cxx + ['--version']),
        'driver': command_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                  '--format=csv,noheader']),
        'cutlass': cutlass_identity(os.environ.get(include_env)) if include_env else None,
        'build_flags': config['build'],
    }


def comparison_identity(environment):
    """Ignore installation paths; freeze dependency identities and the timing device."""
    if not environment:
        raise ValueError('Missing measured environment; qualify a new root')
    return {
        **{key: environment.get(key) for key in
           ('python', 'torch', 'cuda', 'device', 'device_uuid', 'capability', 'packages', 'build_flags')},
        'baseline_modules': {name: info.get('sha256') if info else None
                             for name, info in environment.get('baseline_modules', {}).items()},
        'nvcc': environment.get('nvcc', {}).get('output'),
        'cxx': environment.get('cxx', {}).get('output'),
        'driver': environment.get('driver', {}).get('output'),
        'cutlass': (environment.get('cutlass') or {}).get('header_tree_sha256'),
    }
