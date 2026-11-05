"""
Merge per-batch pandera failure cases into one whole-file report.

The inventory upload is validated batch by batch; ``FailureCases`` combines the
batches' failures so the error count and the first 100 cases match what
validating the whole file at once would report.
"""

import numpy as np
import pandas as pd

MAX_REPORTED_CASES = 100

_KEY = ["schema_context", "column", "check", "check_number"]
_UNIQUENESS_KEY = ("Column", "tree_id", "field_uniqueness", "None")


class FailureCases:
    """Pandera failure cases gathered across batches.

    Pandera lists one block of cases per failed check, ordered schema-level
    first, then by column, then check; scalar (index-less) cases come after the
    row cases. Each (context, column, check) group here keeps its count and
    first 100 cases, and ``report`` orders the groups the same way. Scalar
    cases, such as a missing column, repeat in every batch, so each distinct
    value is counted once.

    Args:
        schema_columns: The schema's column names, in schema order.
    """

    def __init__(self, schema_columns: list[str]):
        self._schema_columns = schema_columns
        self._frame_columns: list[str] = []
        self._groups: dict[tuple, dict] = {}

    def __bool__(self) -> bool:
        return bool(self._groups)

    def add(self, failure_cases: pd.DataFrame, frame_columns: list[str]) -> None:
        """Add one batch's ``SchemaErrors.failure_cases``.

        Args:
            failure_cases: The batch's failure cases.
            frame_columns: Columns of the validated frame, in frame order.
        """
        self._frame_columns = self._frame_columns or frame_columns
        fc = failure_cases.reset_index(drop=True)
        # str() per value, so a None check_number keys as "None", not NaN.
        keys = fc[_KEY].map(str)
        for key, cases in fc.groupby([keys[c] for c in _KEY], sort=False, dropna=False):
            scalar = bool(cases["index"].isna().all())
            group = self._groups.setdefault(
                key, {"count": 0, "cases": [], "seen": set(), "scalar": scalar}
            )
            if scalar:
                for case in cases.to_dict("records"):
                    if repr(case["failure_case"]) not in group["seen"]:
                        group["seen"].add(repr(case["failure_case"]))
                        group["count"] += 1
                        if len(group["cases"]) < MAX_REPORTED_CASES:
                            group["cases"].append(case)
            else:
                group["count"] += len(cases)
                room = MAX_REPORTED_CASES - len(group["cases"])
                group["cases"].extend(cases.head(room).to_dict("records"))

    def add_duplicate_tree_ids(self, blocks: list[tuple[int, np.ndarray]]) -> None:
        """Report every occurrence of a repeated ``tree_id``, in row order.

        Args:
            blocks: ``(first row index, tree_ids)`` per batch, in file order.
        """
        if not blocks:
            return
        ids = np.sort(np.concatenate([b for _, b in blocks]))
        repeated = np.unique(ids[1:][ids[1:] == ids[:-1]])
        if len(repeated) == 0:
            return
        rows = np.concatenate(
            [start + np.flatnonzero(np.isin(b, repeated)) for start, b in blocks]
        )
        values = np.concatenate([b[np.isin(b, repeated)] for _, b in blocks])
        self._groups[_UNIQUENESS_KEY] = {
            "count": len(rows),
            "cases": [
                {
                    "schema_context": "Column",
                    "column": "tree_id",
                    "check": "field_uniqueness",
                    "check_number": None,
                    "failure_case": int(v),
                    "index": int(i),
                }
                for v, i in zip(values[:MAX_REPORTED_CASES], rows[:MAX_REPORTED_CASES])
            ],
            "seen": set(),
            "scalar": False,
        }

    def report(self) -> tuple[int, list[dict]]:
        """Return the total failure count and the first 100 cases."""
        total = sum(g["count"] for g in self._groups.values())
        cases = []
        for key in sorted(self._groups, key=self._rank):
            cases.extend(self._groups[key]["cases"])
            if len(cases) >= MAX_REPORTED_CASES:
                break
        return total, cases[:MAX_REPORTED_CASES]

    def _rank(self, key: tuple) -> tuple:
        context, column, check, number = key
        scalar = self._groups[key]["scalar"]
        number = -1 if number in ("None", "nan") else int(float(number))
        if context == "DataFrameSchema":
            phase = 0 if check == "column_in_dataframe" else 1
            position = (
                self._frame_columns.index(column)
                if column in self._frame_columns
                else -1
            )
            return (0, scalar, phase, position, number)
        if check.startswith("coerce_dtype"):
            phase, order = 0, 0
        else:
            phase = 1
            order = {"not_nullable": 0, "field_uniqueness": 1}.get(
                check, 2 if check.startswith("dtype(") else 3 + number
            )
        position = (
            self._schema_columns.index(column)
            if column in self._schema_columns
            else len(self._schema_columns)
        )
        return (1, scalar, phase, position, order)
