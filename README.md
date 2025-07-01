ollama codes

sources:

- https://dev.to/apilover/ollama-cheatsheet-running-llms-locally-with-ollama-j84

-

* list the models pulled from Hugging Face

  ```bash
  ollama list
  ```

* How to check if ollama server is running in the background. Open web browser and check

```
     http://127.0.0.1:11434
```

Should see message that `Ollama is running`

- To Pull a Model

  ```bash
  ollama pull qwen2:7b #here qwen2:7b is the model name
  ```

- To remove a Model

  ```bash
  ollama rm qwen2:7b
  ```

---

- create Hugging Face account

  - Create a Hugging Face Access token
    - go [here](https://huggingface.co/settings/tokens)
    - Select create new token

Create Langfuse account -
Allows you to perform `Observability` (i.e., traceability) for monitoring and analysis of the Agent with the
assistance of `SmolagentsInstrumentor` which uses the `OpenTelemetry`(https://opentelemetry.io/) standard for instrumenting agent runs. Helps with inspections and logging.

- create a Langfuse project so that you can create two API tokens

  - go [here](https://us.cloud.langfuse.com)
  - Create a new project which will allow you to generate
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY

---

tools

`Langfuse` open-source LLM (Large Language Model) engineering platform designed to help develop, monitor, evaluate, and debug AI applications.

`OpenTelemetry` an open-source observability framework to collect, process, and export telemetry data (metrics, logs, and traces)

`smolagents` Agentic Framework to help create Agents, CodeAgents and Tool-Calling specifically,

`LlamaIndex` Agentic Framework to help create Agents,

`LangGraph` Agentic Framework to help create Agents,

`LangChain` to support the creation of agents to use LLM reasoning engines

`LLM` to act as reasoning agents for action determination for agents,

`RAG` - for Agentic RAG creatio to combine autonomous agents with dynamic knowledge retrieval
