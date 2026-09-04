package com.apc.modelclient;

import org.springframework.stereotype.Component;

@Component
public class MockClient implements BaseModelClient {
    @Override
    public String complete(String prompt, double temperature) {
        if (prompt.contains("\"city\"") || prompt.contains("city")) {
            return "{\"city\": \"Beijing\", \"population\": 21540000}";
        }
        if (prompt.contains("合同编号")) return "HT-2026-0917";
        if (prompt.contains("上海Q2")) return "135";
        if (prompt.contains("打 8 折")) return "140";
        if (prompt.contains("所有 A 都是 C")) return "不能";
        if (prompt.contains("是否有风险")) return "有";
        if (prompt.contains("只输出 JSON")) return "{\"name\": \"test\", \"age\": 30, \"active\": true}";
        return "{\"answer\": 42}";
    }
    @Override public String getModelId() { return "mock"; }
}
