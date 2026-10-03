"""Evaluates Clef against a CSV of labelled examples.

CSV columns (header names are case-insensitive):

    context   The state the model reasons about.
    question  The question to answer.
    answer    The correct answer.
    options   Optional. Candidate answers separated by any punctuation, such
              as "|", ";" or "/". When empty or the column is missing, the
              options are the distinct answers given for the same question
              anywhere in the file.

Columns can be separated by any punctuation or a tab (rows copied from a
spreadsheet). Common separators are tried first, and the first one that yields
the required headers is used.
"""

import io
import warnings
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("context", "question", "answer")
# Separators to try first; any other punctuation found is tried after these.
COMMON_SEPARATORS = (",", "\t", ";", "|", "\n", "/")
MAX_ROWS = 10_000
MAX_ERRORS_SHOWN = 20
# Clef accepts at most 255 options per choice question.
MAX_OPTIONS = 255

# Upper edges of the confidence bands that accuracy is reported for.
CONFIDENCE_BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]


class CsvError(ValueError):
    pass


@dataclass
class EvalRow:
    # Numbered as a spreadsheet shows it: the header is row 1.
    row_number: int
    context: str
    question: str
    options: list[str]
    answer: str


def _candidate_separators(text: str, exclude: str = "") -> list[str]:
    """Common separators present in `text`, then any other punctuation in it."""
    found = [s for s in COMMON_SEPARATORS if s in text]
    for ch in text:
        if not (ch.isalnum() or ch.isspace() or ch in found or ch in exclude):
            found.append(ch)
    return found


def _split_options(raw: str, answer: str) -> list[str]:
    """Splits an options cell on whichever separator it uses.

    Prefers a split that has the answer as one of its options, so that a
    character inside an answer, like the comma in "Yes, definitely", is not
    taken for the separator.
    """
    splits = []
    for sep in _candidate_separators(raw):
        parts = [o.strip() for o in raw.split(sep) if o.strip()]
        if len(parts) >= 2:
            if answer.casefold() in (p.casefold() for p in parts):
                return parts
            splits.append(parts)
    # No split contains the answer; the first one makes the error message useful.
    if splits:
        return splits[0]
    return [raw.strip()] if raw.strip() else []


def _read(text: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        # Without this, pandas turns the leading fields of rows with extra
        # fields into an index. With it, extra fields are dropped.
        index_col=False,
    )


def _read_table(text: str) -> tuple[pd.DataFrame, bool]:
    """Reads the text with the column separator that yields the required headers.

    Also returns whether some rows had more fields than the header.
    """
    header = text.split("\n", 1)[0]
    tried = []
    for sep in _candidate_separators(header, exclude='"'):
        try:
            columns = {str(c).strip().lower() for c in _read(header, sep).columns}
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        tried.append(sep)
        if set(REQUIRED_COLUMNS) <= columns:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", pd.errors.ParserWarning)
                    df = _read(text, sep)
            except pd.errors.ParserError as e:
                raise CsvError(f"cannot read the file: {e}")
            extra_fields = any(issubclass(w.category, pd.errors.ParserWarning) for w in caught)
            return df, extra_fields
    shown = ", ".join(repr(s) for s in tried) or "none found"
    raise CsvError(
        f"no separator gives the columns {', '.join(REQUIRED_COLUMNS)}. "
        f"Header: {header.strip()[:200]!r}. Separators tried: {shown}."
    )


def parse_csv(text: str) -> list[EvalRow]:
    text = text.lstrip("\ufeff")
    if not text.strip():
        raise CsvError("the file is empty")
    df, extra_fields = _read_table(text)
    columns = {str(c).strip().lower(): c for c in df.columns}
    if len(df) > MAX_ROWS:
        raise CsvError(f"too many rows, the limit is {MAX_ROWS}")

    def column(name: str) -> pd.Series:
        if name not in columns:
            return pd.Series("", index=df.index)
        return df[columns[name]].str.strip()

    table = pd.DataFrame({c: column(c) for c in (*REQUIRED_COLUMNS, "options")})
    table["row_number"] = table.index + 2
    table = table[(table[[*REQUIRED_COLUMNS, "options"]] != "").any(axis=1)]
    if table.empty:
        raise CsvError("the file has a header but no rows")

    # Options for rows that do not list their own: the distinct answers given
    # to the same question, in order of first appearance, ignoring case.
    derived = (
        table[table["answer"] != ""]
        .assign(key=lambda t: t["answer"].str.casefold())
        .drop_duplicates(["question", "key"])
        .groupby("question", sort=False)["answer"]
        .agg(list)
        .to_dict()
    )

    rows, errors = [], []
    for v in table.to_dict("records"):
        problem = None
        options = _split_options(v["options"], v["answer"]) or derived.get(v["question"], [])
        folded = [o.casefold() for o in options]
        if not v["context"]:
            problem = "context is empty"
        elif not v["question"]:
            problem = "question is empty"
        elif not v["answer"]:
            problem = "answer is empty"
        elif len(options) < 2:
            problem = (
                "needs at least 2 options: add an options column, "
                "or more rows with other answers to the same question"
            )
        elif len(options) > MAX_OPTIONS:
            problem = f"has {len(options)} options, the limit is {MAX_OPTIONS}"
        elif len(set(folded)) != len(folded):
            problem = "options contain duplicates"
        elif v["answer"].casefold() not in folded:
            problem = f"answer {v['answer']!r} is not one of the options {options}"
        if problem:
            errors.append(f"row {v['row_number']}: {problem}")
            continue
        # Use the option's spelling, so comparing with the prediction is exact.
        answer = options[folded.index(v["answer"].casefold())]
        rows.append(EvalRow(v["row_number"], v["context"], v["question"], options, answer))

    if errors:
        shown = errors[:MAX_ERRORS_SHOWN]
        more = len(errors) - len(shown)
        hint = (
            "\nSome rows have more fields than the header. Put double quotes around "
            "fields that contain the separator."
            if extra_fields
            else ""
        )
        raise CsvError(
            f"{len(errors)} invalid row(s):\n" + "\n".join(shown)
            + (f"\n…and {more} more" if more else "")
            + hint
        )
    return rows


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> dict:
    """Summarises row results that have `question`, `correct`, `confidence`, or `error`."""
    scored = [r for r in results if r.get("error") is None]
    correct = [r for r in scored if r["correct"]]
    wrong = [r for r in scored if not r["correct"]]

    by_question: dict[str, list[dict]] = {}
    for r in scored:
        by_question.setdefault(r["question"], []).append(r)
    per_question = sorted(
        (
            {
                "question": q,
                "total": len(rs),
                "correct": sum(r["correct"] for r in rs),
                "accuracy": _accuracy(sum(r["correct"] for r in rs), len(rs)),
            }
            for q, rs in by_question.items()
        ),
        key=lambda x: (x["accuracy"], -x["total"]),
    )

    bands = []
    for i, (lo, hi) in enumerate(CONFIDENCE_BANDS):
        last = i == len(CONFIDENCE_BANDS) - 1
        rs = [r for r in scored if lo <= r["confidence"] < hi or (last and r["confidence"] == hi)]
        bands.append(
            {
                "low": lo,
                "high": hi,
                "total": len(rs),
                "correct": sum(r["correct"] for r in rs),
                "accuracy": _accuracy(sum(r["correct"] for r in rs), len(rs)),
            }
        )

    return {
        "total": len(results),
        "scored": len(scored),
        "correct": len(correct),
        "errors": len(results) - len(scored),
        "accuracy": _accuracy(len(correct), len(scored)),
        "mean_confidence_correct": _mean([r["confidence"] for r in correct]),
        "mean_confidence_wrong": _mean([r["confidence"] for r in wrong]),
        "per_question": per_question,
        "confidence_bands": bands,
    }
