"""Raw data sources: expected files and how to fetch them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


@dataclass(frozen=True, slots=True)
class AmlWorldFiles:
    """The three files of one AMLworld variant (default: HI-Small)."""

    transactions: Path
    accounts: Path
    patterns: Path

    @classmethod
    def in_dir(cls, raw_dir: Path, variant: str = "HI-Small") -> AmlWorldFiles:
        return cls(
            transactions=raw_dir / f"{variant}_Trans.csv",
            accounts=raw_dir / f"{variant}_accounts.csv",
            patterns=raw_dir / f"{variant}_Patterns.txt",
        )

    def missing(self) -> list[Path]:
        return [
            path for path in (self.transactions, self.accounts, self.patterns) if not path.is_file()
        ]


class MissingRawDataError(FileNotFoundError):
    """Raised when raw files are absent. The message says exactly how to download them."""

    def __init__(self, missing: list[Path], raw_dir: Path) -> None:
        names = ", ".join(path.name for path in missing)
        super().__init__(
            f"Missing raw AMLworld files in {raw_dir}: {names}\n"
            "Download them (public, no account needed) with:\n"
            "  finint data download"
        )
        self.missing = missing
