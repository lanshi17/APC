package com.apc.modelclient;

public interface BaseModelClient {
    String complete(String prompt, double temperature);
    String getModelId();
}
