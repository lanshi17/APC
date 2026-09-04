package com.apc.modelclient;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.stereotype.Component;

@Component
public class SpringAiClient implements BaseModelClient {

    private final ChatClient chatClient;

    public SpringAiClient(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    @Override
    public String complete(String prompt, double temperature) {
        return chatClient.prompt()
                .user(prompt)
                .options(ChatOptions.builder().temperature(temperature).build())
                .call()
                .content();
    }

    @Override public String getModelId() { return "spring-ai-openai"; }
}
