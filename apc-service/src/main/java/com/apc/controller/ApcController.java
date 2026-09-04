package com.apc.controller;

import com.apc.compiler.PromptRenderer;
import com.apc.domain.*;
import com.apc.modelclient.MockClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class ApcController {

    private final PromptRenderer renderer;
    private final MockClient mockClient;

    public ApcController(PromptRenderer renderer, MockClient mockClient) {
        this.renderer = renderer;
        this.mockClient = mockClient;
    }

    @GetMapping("/health")
    public Map<String, String> health() { return Map.of("status", "UP", "service", "apc-service"); }

    @PostMapping("/compile")
    public ResponseEntity<CompiledPrompt> compile(@RequestBody CompileRequest req) {
        CompiledPrompt cp = renderer.compile(req.genome(), req.taskSpec(), req.profile());
        return ResponseEntity.ok(cp);
    }

    @PostMapping("/probe/mock")
    public Map<String, Object> mockProbe(@RequestBody Map<String, String> body) {
        String prompt = body.getOrDefault("prompt", "hello");
        String out = mockClient.complete(prompt, 0.0);
        return Map.of("modelId", "mock", "response", out);
    }

    public record CompileRequest(TaskSpec taskSpec, PromptGenome genome, ModelProfile profile) {}
}
