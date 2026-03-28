# GenAI Intervention Module

This module turns the current risk engine into an intervention-draft engine.

## What is implemented

- A context builder that assembles a `VECTOR Context Packet`
- A prompt builder for an external GenAI provider
- A deterministic fallback draft generator
- A new API endpoint that returns:
  - latest risk + physics + decision matrix context
  - SHAP driver summary
  - LLM-ready prompt
  - draft intervention message

## Current endpoint

- `GET /intervention-report/{account_id}`

This endpoint does not call an external LLM yet.
It prepares the full context and returns a draft message that can later be replaced by OpenAI, Anthropic, Gemini, or a local model.

## Recommended API choice

### Free prototype

Use a local model via Ollama.

- Cost: free on your machine
- Good enough for UI and prompt-flow prototyping
- Best if you want to avoid API billing at the start

Suggested local models:

- `llama3.1:8b`
- `mistral`
- `qwen2.5`

### Hosted prototype

Use the OpenAI Responses API.

Why:

- OpenAI recommends the Responses API for new projects
- It supports structured outputs and tool use cleanly

Official docs:

- https://platform.openai.com/docs/api-reference/responses/create
- https://platform.openai.com/docs/guides/migrate-to-responses

Pricing:

- Responses API is billed at the chosen model's token rates
- Official pricing: https://openai.com/api/pricing

Practical recommendation:

- cheapest hosted prototype: `gpt-5 mini` or `gpt-4.1 mini`
- stronger quality prototype: `gpt-5.4 mini`

## Why not wire the provider immediately

The intervention context and prompt structure should stabilize first.
That lets you:

- test prompt quality from stored risk payloads
- review drafts with humans
- avoid coupling your architecture to one provider too early

## Next step

Add a provider adapter:

- `OpenAIInterventionProvider`
- `OllamaInterventionProvider`

Both should consume the `llm_prompt` returned by `/intervention-report/{account_id}`.
