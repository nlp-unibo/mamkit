"""Local entry point for checks run by GitHub Actions."""

from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
LINT_VERSION = "3.12"
TORCH = ["torch>=2.3.0", "torchaudio>=2.3.0"]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.error_on_missing_interpreters = False
nox.options.sessions = ["lint", "tests"]


def install(session: nox.Session, *args: str) -> None:
    # CPU wheels first: no check needs the CUDA libraries the default wheels pull in.
    session.install(*TORCH, "--index-url", TORCH_CPU_INDEX)
    session.install(*args)


@nox.session(python=LINT_VERSION)
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.15.20")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    install(session, "-e", ".[dev]")
    session.run(
        "pytest",
        "--cov=mamkit",
        "--cov-branch",
        # Measured at 77.21% on Python 3.10 on 2026-11-01, with the rewritten
        # loaders tested. The legacy modules are removed in #16.
        "--cov-fail-under=77",
        "--cov-report=term-missing",
    )


@nox.session(python=LINT_VERSION)
def docs(session: nox.Session) -> None:
    install(session, "-e", ".[docs]")
    session.chdir("docsrc")
    session.run("bash", "build_docs.sh", external=True)


@nox.session(python=LINT_VERSION)
def package(session: nox.Session) -> None:
    session.install("build", "twine")
    session.run(
        "python",
        "-c",
        "import shutil; shutil.rmtree('dist', ignore_errors=True)",
    )
    session.run("python", "-m", "build")
    artifacts = [path.resolve() for path in Path("dist").glob("*")]
    session.run("twine", "check", "--strict", *(str(path) for path in artifacts))
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    install(session, str(wheel))
    session.chdir("/tmp")
    session.run("python", "-c", "import mamkit; print(mamkit.__version__)")
