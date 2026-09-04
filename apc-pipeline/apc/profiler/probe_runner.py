from __future__ import annotations
import json, yaml, pathlib
from apc.models.base import BaseModelClient

class ProbeRunner:
    def __init__(self, model_client: BaseModelClient): self.model_client = model_client
    def run_probe(self, probe: dict) -> dict:
        response = self.model_client.complete(probe["input"]["prompt"], temperature=0.0)
        metrics = self.score_probe_response(probe, response)
        return {"probe_id": probe["probe_id"], "category": probe["category"], "response": response, "metrics": metrics}
    def run_suite(self, suite_dir: str) -> list[dict]:
        results = []
        for p in sorted(pathlib.Path(suite_dir).glob("*.yaml")):
            with open(p, encoding="utf-8") as f: probe = yaml.safe_load(f)
            results.append(self.run_probe(probe))
        return results
    def score_probe_response(self, probe: dict, response: str) -> dict:
        m = {}
        exp = probe.get("expected", {})
        if exp.get("json_parsable"): m["json_parse_success"] = self.check_json_parsable(response)
        if "required_fields" in exp: m["required_fields_present"] = self.check_required_fields(response, exp["required_fields"])
        if "answer" in exp: m["exact_match"] = response.strip() == exp["answer"].strip()
        if "contains_any" in exp: m["contains_any"] = any(k in response for k in exp["contains_any"])
        if "forbidden" in exp: m["no_forbidden"] = all(k not in response for k in exp["forbidden"])
        return m
    def check_json_parsable(self, r: str) -> bool:
        try: json.loads(r); return True
        except: return False
    def check_required_fields(self, r: str, fields: list[str]) -> bool:
        try: return all(f in json.loads(r) for f in fields)
        except: return False
