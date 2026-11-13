"""Data classification for LLM context (task 1.9 · R1.7)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from finint.config import load_yaml
from finint.data.classification import DataClassification, UnclassifiedColumnsError

CONFIG = Path(__file__).parents[1] / "config"
LABELS = ("is_laundering", "attempt_id", "typology")


@pytest.fixture(scope="module")
def classification() -> DataClassification:
    return DataClassification(load_yaml("data_classification", CONFIG))


@pytest.mark.parametrize("column", LABELS)
def test_label_columns_are_forbidden(classification: DataClassification, column: str) -> None:
    assert classification.level("transactions", column) == "forbidden"
    assert column not in classification.llm_visible("transactions")


@pytest.mark.parametrize(
    "table", ["injected_sanctions", "injected_media", "injected_prompts", "attempts"]
)
def test_ground_truth_tables_are_fully_forbidden(
    classification: DataClassification, table: str
) -> None:
    assert classification.level(table, "anything") == "forbidden"
    assert classification.llm_visible(table) == []


def test_memo_and_names_are_untrusted(classification: DataClassification) -> None:
    assert classification.level("transactions", "reference") == "untrusted"
    assert classification.level("entities", "synthetic_name") == "untrusted"


def test_enforce_rejects_unclassified_columns(
    classification: DataClassification, tmp_path: Path
) -> None:
    db = tmp_path / "x.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE transactions (txn_id BIGINT, secret_new_column VARCHAR)")
        con.execute("CREATE TABLE brand_new_table (a INT)")
    with pytest.raises(UnclassifiedColumnsError) as excinfo:
        classification.enforce(db)
    assert "transactions.secret_new_column" in str(excinfo.value)
    assert "brand_new_table.a" in str(excinfo.value)
    assert "transactions.txn_id" not in str(excinfo.value)


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        DataClassification({"tables": {"t": {"c": "public"}}})
