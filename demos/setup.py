from pathlib import Path
from cinnamon.registry import RegistrationKey, Registry

if __name__ == '__main__':
    Registry.setup(directory=Path('.').resolve().parent)

