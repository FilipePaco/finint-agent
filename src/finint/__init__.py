"""finint-agent: agentic AML alert investigation with ML scoring."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("finint-agent")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
