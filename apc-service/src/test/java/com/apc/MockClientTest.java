package com.apc;

import com.apc.modelclient.MockClient;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class MockClientTest {

    private final MockClient client = new MockClient();

    @Test
    void probeShortcutsAnswerDeterministically() {
        assertThat(client.complete("请查合同编号是多少", 0.0)).isEqualTo("HT-2026-0917");
        assertThat(client.complete("上海Q2营收", 0.0)).isEqualTo("135");
    }

    @Test
    void jsonHintReturnsParsableObject() {
        assertThat(client.complete("请只输出 JSON 结果", 0.0)).contains("\"name\"");
    }

    @Test
    void defaultAnswerIsStable() {
        assertThat(client.complete("今天天气怎么样", 0.0)).isEqualTo("{\"answer\": 42}");
        assertThat(client.getModelId()).isEqualTo("mock");
    }
}
