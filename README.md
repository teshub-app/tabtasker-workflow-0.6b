# TabTasker Workflow AI (0.6B, ONNX)

> Apache-2.0 &bull; base model [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) &bull; ONNX / q4 &bull; runs in the browser via [Transformers.js](https://github.com/huggingface/transformers.js)

The on-device workflow generator behind **[TabTasker](https://tabtasker.com)**, a free,
browser-based toolbox where files are processed locally and never uploaded to a server.

Describe what you want in plain English (for example, "Take a PDF, pull out the text, and
read it aloud as an MP3") and this model replies with a complete, valid TabTasker workflow
graph as JSON: which node types to use, how to wire their ports, and what inputs to expose.
It powers the **[Create with AI](https://tabtasker.com/workflow)** feature and runs entirely
in your browser via WebGPU (with a CPU/wasm fallback) using
[Transformers.js](https://github.com/huggingface/transformers.js). One download, zero server
calls, full privacy.

Try it live at **[tabtasker.com/workflows](https://tabtasker.com/workflows)**.

## What it does

Given TabTasker's node catalog in the system prompt, the model emits one JSON object:

```json
{"schemaVersion":1,"name":"PDF to audio","nodes":[
  {"id":"n1","type":"pdf-upload","position":[60,100],"config":{}},
  {"id":"n2","type":"pdf-to-text","position":[300,100],"config":{}},
  {"id":"n3","type":"text-speech","position":[540,100],"config":{}}],
 "edges":[
  {"fromNode":"n1","fromPort":"out","toNode":"n2","toPort":"in"},
  {"fromNode":"n2","fromPort":"out","toNode":"n3","toPort":"in"}],
 "inputs":[]}
```

Every port connection is type checked (pdf to text to audio above), and TabTasker validates
the graph with its real graph validator before anything runs.

## Base model and attribution

This model is a fine-tune of **[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** by
the **Qwen team at Alibaba Cloud**, released under the Apache License 2.0. We thank the Qwen
team for making strong small models openly available. This project would not exist without
their work.

```bibtex
@misc{qwen3,
  title  = {Qwen3 Technical Report},
  author = {Qwen Team},
  year   = {2025},
  url    = {https://huggingface.co/Qwen/Qwen3-0.6B}
}
```

## How it was trained

- **Base model:** Qwen3-0.6B (about 600M parameters)
- **Method:** LoRA (r=16, alpha=32, all attention and MLP projections), merged to 16-bit
  after training. Trained with [Unsloth](https://github.com/unslothai/unsloth) and TRL on a
  single RTX 5090.
- **Data:** about 3,400 request-to-workflow-JSON pairs generated from TabTasker's live node
  catalog: the real product templates, hand-authored task archetypes, and bounded synthetic
  node chains. Every training graph passes the app's graph validator by construction. Splits
  are per-graph, so no graph shape is shared between train and test.
- **Prompting:** trained byte-identical to the app's inference prompt (system prompt with the
  node catalog plus /no_think), so there is no train and serve skew.
- **Quantization:** ONNX export via optimum, quantized with the Transformers.js pipeline. The
  app ships the q4 variant (4-bit MatMul weights, fp32 activations). fp16 activations overflow
  on this model family, so q4 is the shipped build.

## Evaluation

Scored with TabTasker's actual graph validator (structural validity) against held-out data.
"Unseen graphs" means graph shapes never seen in training. "Held-out phrasings" means known
graphs with new wording.

| Metric | Base Qwen3-0.6B | Fine-tuned (bf16) | Fine-tuned (q4, shipped) |
|---|---|---|---|
| Valid JSON (unseen graphs) | 18.1% | 99.5% | 100% |
| Valid workflow graph | 3.5% | 96.0% | 83.9% |
| Node-type F1 | 0.30 | 0.94 | 0.92 |
| Edge F1 | 0.03 | 0.91 | 0.85 |
| Exact graph match | 0% | 85.4% | 71.9% |
| Valid graph (held-out phrasings) | 20.0% | 97.0% | 86.0% |

4-bit quantization costs about 12 points of first-shot graph validity on this small model.
TabTasker compensates in-app: every generation runs through the real graph validator, and an
invalid draft triggers one automatic regeneration, putting effective success around 97%.

## Use with Transformers.js

```js
import { pipeline, env } from '@huggingface/transformers';

// TabTasker serves this model from its own CDN.
env.allowLocalModels = false;
env.remoteHost = 'https://models.tabtasker.com/';
env.remotePathTemplate = '{model}/';

const generate = await pipeline('text-generation', 'tabtasker-workflow-0.6b', {
  dtype: 'q4',
  device: 'webgpu', // falls back to 'wasm' where WebGPU is unavailable
});

const messages = [
  { role: 'system', content: SYSTEM_PROMPT_WITH_NODE_CATALOG + '\n/no_think' },
  { role: 'user', content: 'Turn a PDF into an MP3 audiobook.' },
];
const out = await generate(messages, { max_new_tokens: 1024, do_sample: false });
```

The system prompt must list the TabTasker node catalog in the format the model was trained on.
See it live in the [Create with AI](https://tabtasker.com/workflow) feature, or just use
TabTasker where it is already wired up.

## Files

- `onnx/model_q4.onnx`: 4-bit weights, fp32 activations (what TabTasker ships). No q4f16
  variant, since this model's fp16 activations overflow and silently destroy output quality.
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and the
  other tokenizer files.

## Links

- **TabTasker:** https://tabtasker.com
- **Create with AI (workflows):** https://tabtasker.com/workflows
- **Base model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## License

[Apache License 2.0](./LICENSE), the same as the base model. Copyright for the base weights
remains with the Qwen team and Alibaba Cloud. The fine-tuned weights are released by TabTasker
under the same terms.
