import json
import unittest
from pathlib import Path

import numpy as np

from run_formal_gold_v1 import (
    fedresolve_adapter,
    item_metrics,
    ragroute_adapter,
    score_views,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "formal_gold_v1_protocol.json"
if not PROTOCOL_PATH.is_file():
    PROTOCOL_PATH = ROOT.parent / "configs" / "formal_gold_v1_protocol.json"


class FormalGoldProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        self.inputs = {
            "canonical_id": "FRR-CONTENT-9999",
            "candidate_ids": ["p0", "p1", "p2", "p3"],
            "candidate_clients": [0, 1, 2, 3],
            "candidate_bytes": [100, 100, 100, 100],
            "full_scores": np.asarray([0.80, 0.72, 0.20, 0.10]),
            "slot_scores": np.asarray(
                [
                    [0.90, 0.20, 0.10, 0.05],
                    [0.15, 0.88, 0.10, 0.05],
                ]
            ),
            "image_similarity": np.eye(4),
        }

    def test_fedresolve_selects_complementary_clients(self):
        views = score_views(self.inputs, self.protocol)
        output = fedresolve_adapter(
            self.inputs,
            views,
            np.ones(8, dtype=bool),
            2,
            self.protocol,
        )
        self.assertEqual(set(output["queried_clients"]), {0, 1})
        self.assertFalse(output["abstain"])

    def test_fedresolve_detects_critical_offline_slot(self):
        views = score_views(self.inputs, self.protocol)
        online = np.ones(8, dtype=bool)
        online[0] = False
        output = fedresolve_adapter(
            self.inputs, views, online, 2, self.protocol
        )
        self.assertTrue(output["abstain"])
        self.assertTrue(output["trace"]["critical_missing_slots"])

    def test_ragroute_retrieves_best_page_inside_selected_client(self):
        inputs = dict(self.inputs)
        inputs["candidate_clients"] = [0, 0, 1, 2]
        inputs["full_scores"] = np.asarray([0.10, 0.90, 0.80, 0.70])
        views = score_views(inputs, self.protocol)
        output = ragroute_adapter(
            inputs,
            views,
            np.ones(8, dtype=bool),
            1,
            self.protocol,
        )
        self.assertEqual(output["queried_clients"], [0])
        self.assertEqual(output["evidence_ids"], ["p1"])

    def test_gold_is_evaluation_only(self):
        views = score_views(self.inputs, self.protocol)
        output = fedresolve_adapter(
            self.inputs,
            views,
            np.ones(8, dtype=bool),
            2,
            self.protocol,
        )
        gold = {
            "minimum_sufficient_client_sets": [[0, 1]],
            "candidate_assessments": [
                {"candidate_id": "p0", "stance": "support"},
                {"candidate_id": "p1", "stance": "support"},
                {"candidate_id": "p2", "stance": "unknown"},
                {"candidate_id": "p3", "stance": "unknown"},
            ],
        }
        metrics = item_metrics(output, gold, self.inputs, 512)
        self.assertEqual(metrics["complete_client_set"], 1.0)
        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertEqual(metrics["strict_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
