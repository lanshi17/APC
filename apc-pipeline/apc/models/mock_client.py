from apc.models.base import BaseModelClient

class MockClient(BaseModelClient):
    def __init__(self, model_id: str = "mock"):
        self._model_id = model_id
    @property
    def model_id(self) -> str: return self._model_id
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if '"city"' in prompt or '"population"' in prompt: return '{"city": "Beijing", "population": 21540000}'
        if "合同编号" in prompt: return "HT-2026-0917"
        if "上海Q2" in prompt: return "135"
        if "打 8 折" in prompt: return "140"
        if "所有 A 都是 C" in prompt: return "不能"
        if "是否有风险" in prompt: return "有"
        if "只输出 JSON" in prompt: return '{"name": "test", "age": 30, "active": true}'
        if "合同金额" in prompt and "50" in prompt: return "50万元"
        return '{"answer": 42}'
