

## Prepare
- uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Python
```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

- Dataset
```bash
python convert_to_sharegpt_dataset.py --input vi-en.json --output "data/inputs/sharegpt_vi-en.json" --src vi --tgt en
```

- Benchmark Tool
```bash
git clone https://github.com/furiosa-ai/vllm.git -b add_power_monitor
```

- Inference Server
  - hf token
  ```bash
  huggingface_cli login --token <YOUR_HF_TOKEN>
  ```
  - npu
    ```bash
    furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct --device "npu:0"
    ```
  - gpu
    ```bash
    vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
    ```

## TPS, TPS/W, BLEU
- npu
```bash
./script-tpsw-npu.sh <result_suffix>
```

- gpu
```bash
./script-tpsw-gpu.sh <result_suffix>
```
