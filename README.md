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

- create a Hugging Face Access token

  - got [here] (https://huggingface.co/settings/tokens)
  - Select create new token
    -
