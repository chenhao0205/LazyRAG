from __future__ import annotations

from evo.eval_qc.constants import LOGIC_IDS
from evo.eval_qc.fields import missing_required_fields


def test_logic_ids_contract() -> None:
    assert LOGIC_IDS == ('query_to_gt_answer', 'query_to_gt_text', 'query_to_key_points')


def test_missing_required_fields_lists_empty_fields() -> None:
    missing = missing_required_fields(query='', gt_answer=' ', gt_text=[], key_points=[])
    assert missing == ['query', 'gt_answer', 'gt_text', 'key_points']


def test_missing_required_fields_none_when_complete() -> None:
    assert missing_required_fields(query='q', gt_answer='a', gt_text=['t'], key_points=['k']) == []
