from __future__ import annotations

from evo.eval_qc.constants import LOGIC_IDS

# eval_qc stage 1 (extract): decompose the query into unit/core/qualifier.
EVAL_QC_EXTRACT: dict = {
    'type': 'object',
    'required': ['query'],
    'properties': {
        'query': {
            'type': 'object',
            'required': ['core', 'qualifier'],
            'properties': {
                'unit': {'type': 'string'},
                'core': {'type': 'string', 'minLength': 1},
                'qualifier': {'type': 'string'},
            },
            'additionalProperties': False,
        },
    },
    'additionalProperties': False,
}
# eval_qc stage 2 (judge): one flat judgment per logic_id, each falling into one bucket.
EVAL_QC_JUDGE: dict = {
    'type': 'object',
    'required': ['judgments', 'summary_reason'],
    'additionalProperties': False,
    'properties': {
        'judgments': {
            'type': 'array',
            'minItems': len(LOGIC_IDS),
            'maxItems': len(LOGIC_IDS),
            'items': {
                'type': 'object',
                'required': ['logic_id', 'reason', 'level'],
                'additionalProperties': False,
                'properties': {
                    'logic_id': {'type': 'string', 'enum': list(LOGIC_IDS)},
                    'reason': {'type': 'string', 'minLength': 1},
                    'level': {'enum': [0.1, 0.3, 0.6, 0.9]},
                },
            },
            'allOf': [
                {
                    'contains': {
                        'type': 'object',
                        'required': ['logic_id'],
                        'properties': {'logic_id': {'const': logic_id}},
                    },
                    'minContains': 1,
                    'maxContains': 1,
                }
                for logic_id in LOGIC_IDS
            ],
        },
        'summary_reason': {'type': 'string'},
    },
}

__all__ = ['EVAL_QC_EXTRACT', 'EVAL_QC_JUDGE']
