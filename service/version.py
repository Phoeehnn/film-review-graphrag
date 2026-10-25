import hashlib
from pathlib import Path

SERVICE_VERSION='1.4.1'


def runtime_source_hashes():
    directory=Path(__file__).parent
    return {'service/'+path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*directory.glob('*.py'),*directory.glob('*.json')])}
