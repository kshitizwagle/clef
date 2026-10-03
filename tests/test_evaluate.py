import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from clef.app import create_app
from clef.evaluate import CsvError, parse_csv, summarize


def test_parses_explicit_options_and_normalises_answer_case():
    rows = parse_csv(
        "Context,Question,Answer,Options\n"
        "c1,Which team?,TECHNICAL,billing| technical |shipping\n"
    )
    assert len(rows) == 1
    assert rows[0].options == ["billing", "technical", "shipping"]
    assert rows[0].answer == "technical"
    assert rows[0].row_number == 2


def test_derives_options_from_answers_to_the_same_question():
    rows = parse_csv(
        "context,question,answer\n"
        "c1,Down?,yes\n"
        "c2,Down?,no\n"
        "c3,Down?,Yes\n"
    )
    assert [r.options for r in rows] == [["yes", "no"]] * 3
    assert rows[2].answer == "yes"


def test_skips_blank_rows_and_strips_bom():
    rows = parse_csv("\ufeffcontext,question,answer,options\n,,,\nc,q,a,a|b\n")
    assert len(rows) == 1


def test_reports_missing_columns():
    with pytest.raises(CsvError, match="no separator gives the columns"):
        parse_csv("context,question\nc,q\n")


def test_reports_every_invalid_row_with_its_row_number():
    with pytest.raises(CsvError) as e:
        parse_csv(
            "context,question,answer,options\n"
            "c,q,maybe,yes|no\n"      # answer not an option
            ",q,yes,yes|no\n"         # empty context
            "c,lonely,only,\n"        # one distinct answer, no options
            "c,q,a,a|A\n"             # duplicate options
        )
    msg = str(e.value)
    assert "4 invalid row(s)" in msg
    assert "row 2: answer 'maybe' is not one of the options" in msg
    assert "row 3: context is empty" in msg
    assert "row 4: needs at least 2 options" in msg
    assert "row 5: options contain duplicates" in msg


def test_rejects_header_only():
    with pytest.raises(CsvError, match="no rows"):
        parse_csv("context,question,answer\n")


def test_summarize():
    s = summarize(
        [
            {"question": "a", "correct": True, "confidence": 0.99, "error": None},
            {"question": "a", "correct": False, "confidence": 0.6, "error": None},
            {"question": "b", "correct": True, "confidence": 0.3, "error": None},
            {"question": "b", "correct": None, "confidence": None, "error": "boom"},
        ]
    )
    assert (s["total"], s["scored"], s["correct"], s["errors"]) == (4, 3, 2, 1)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["mean_confidence_correct"] == pytest.approx(0.645)
    assert s["mean_confidence_wrong"] == pytest.approx(0.6)
    assert [q["question"] for q in s["per_question"]] == ["a", "b"]  # worst first
    assert [b["total"] for b in s["confidence_bands"]] == [1, 1, 0, 1]


def fake_upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body["state"] == "upstream fails":
        return httpx.Response(500, text="boom")
    options = list(body["questions"]["q0"]["criteria"])
    # Always predicts the first option.
    probs = {o: (0.9 if i == 0 else 0.1 / (len(options) - 1)) for i, o in enumerate(options)}
    return httpx.Response(
        200,
        json={"answers": {"q0": {"type": "choice", "choice": options[0],
                                 "probabilities": probs, "confidence": 0.8}}},
    )


client = TestClient(create_app(transport=httpx.MockTransport(fake_upstream), manage_server=False))


def test_evaluate_streams_rows_and_summary():
    csv_text = (
        "context,question,answer,options\n"
        "c1,Which team?,billing,billing|technical\n"
        "c2,Which team?,technical,billing|technical\n"
        "upstream fails,Which team?,billing,billing|technical\n"
    )
    r = client.post("/api/evaluate", content=csv_text, headers={"content-type": "text/csv"})
    assert r.status_code == 200
    messages = [json.loads(line) for line in r.text.splitlines()]
    assert messages[0] == {"type": "start", "total": 3}
    rows = {m["row_number"]: m for m in messages if m["type"] == "row"}
    assert rows[2]["correct"] is True and rows[2]["answer_probability"] == pytest.approx(0.9)
    assert rows[3]["correct"] is False and rows[3]["prediction"] == "billing"
    assert rows[4]["error"].startswith("model server returned 500")
    summary = messages[-1]
    assert summary["type"] == "summary"
    assert (summary["correct"], summary["scored"], summary["errors"]) == (1, 2, 1)


def test_evaluate_returns_400_with_row_errors():
    r = client.post("/api/evaluate", content="context,question,answer\nc,q,a\n")
    assert r.status_code == 400
    assert "row 2: needs at least 2 options" in r.json()["detail"]


def test_sample_csv_is_valid():
    r = client.get("/static/sample.csv")
    assert r.status_code == 200
    assert len(parse_csv(r.text)) == 9


def test_accepts_tab_separated_text_from_a_spreadsheet():
    rows = parse_csv(
        "context\tquestion\tanswer\toptions\n"
        "Paid twice, refund please, now.\tWhich team?\tbilling\tbilling|technical\n"
    )
    assert rows[0].context == "Paid twice, refund please, now."
    assert rows[0].options == ["billing", "technical"]


def test_semicolon_separated_options():
    rows = parse_csv(
        "context,question,answer,options\n"
        "c,Is it true?,NOT GIVEN,TRUE;FALSE;NOT GIVEN\n"
    )
    assert rows[0].options == ["TRUE", "FALSE", "NOT GIVEN"]
    assert rows[0].answer == "NOT GIVEN"


def test_semicolon_separated_columns_with_pipe_options():
    # European Excel exports use ";" between columns.
    rows = parse_csv("context;question;answer;options\nc, with comma;q;b;a|b\n")
    assert (rows[0].context, rows[0].options) == ("c, with comma", ["a", "b"])


def test_pipe_separated_columns_with_semicolon_options():
    rows = parse_csv("context|question|answer|options\nc;still context?|q|b|a;b\n")
    assert rows[0].options == ["a", "b"]


def test_quoted_fields_with_commas_and_escaped_quotes():
    rows = parse_csv((Path(__file__).parent / "data" / "reading.csv").read_text())
    assert len(rows) == 8
    assert rows[0].options == ["TRUE", "FALSE", "NOT GIVEN"]
    assert rows[5].question == 'The word "icebox" comes from the __________ trade.'
    assert "in winter. The ice was packed in sawdust, stored" in rows[3].context


def test_any_column_separator():
    for sep in ["^", "~", ":", "#"]:
        rows = parse_csv(f"context{sep}question{sep}answer{sep}options\nsome, text{sep}q{sep}b{sep}a|b\n")
        assert (rows[0].context, rows[0].options) == ("some, text", ["a", "b"]), sep


def test_any_option_separator():
    for options in ["yes/no", "yes / no", "yes - no", "yes\nno", "yes\tno", "yes, no"]:
        csv_text = 'context,question,answer,options\nc,q,no,"' + options + '"\n'
        assert parse_csv(csv_text)[0].options == ["yes", "no"], options


def test_option_split_keeps_the_answer_whole():
    # The comma is part of the answer; the options are split on ";".
    rows = parse_csv('context,question,answer,options\nc,q,"Yes, definitely","Yes, definitely;No"\n')
    assert rows[0].options == ["Yes, definitely", "No"]
    assert rows[0].answer == "Yes, definitely"


def test_rows_with_line_breaks_are_numbered_like_a_spreadsheet():
    rows = parse_csv('context,question,answer,options\n"multi\nline",q,a,a|b\nc,q,b,a|b\n')
    assert [r.row_number for r in rows] == [2, 3]


def test_accepts_trailing_separators():
    rows = parse_csv("context,question,answer,options,\nc,q,a,a|b,\n")
    assert rows[0].options == ["a", "b"]


def test_unquoted_separator_in_a_field_is_reported_with_a_hint():
    with pytest.raises(CsvError) as e:
        parse_csv("context,question,answer,options\nfoo, bar,q,a,a|b\nc,q,a,a|b\n")
    msg = str(e.value)
    assert "row 2:" in msg
    assert "row 3" not in msg
    assert "Put double quotes around fields that contain the separator" in msg
