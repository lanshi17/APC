from __future__ import annotations
import json

class RuleBasedChecker:
    def check_json(self, output: str, schema: dict | None) -> dict:
        try: data = json.loads(output)
        except Exception: return {"json_parsable": False, "format_score": 0.0}
        if not schema: return {"json_parsable": True, "format_score": 1.0}
        missing = [k for k in schema.keys() if k not in data]
        extra = [k for k in data.keys() if k not in schema]
        score = 1.0
        if missing: score -= 0.4
        if extra: score -= 0.2
        return {"json_parsable": True, "missing_fields": missing, "extra_fields": extra, "format_score": max(score,0.0)}
