# LLM Inference Frameworks Comparison

## Background

To accelerate inference speed, three commonly used frameworks for
serving LLMs were selected for comparative testing:

------------------------------------------------------------------------

### vLLM

A fast and easy-to-use library for LLM inference and serving.

**Key Features:** - State-of-the-art serving throughput - Efficient
management of attention key and value memory with *PagedAttention* -
Continuous batching of incoming requests - Fast model execution with
CUDA/HIP graph - Quantization: GPTQ, AWQ, SqueezeLLM, FP8 KV Cache -
Optimized CUDA kernels

------------------------------------------------------------------------

### TensorRT-LLM

Provides users with an easy-to-use Python API to define Large Language
Models (LLMs) and build TensorRT engines that contain state-of-the-art
optimizations to perform inference efficiently on NVIDIA GPUs.

**Key Features:** - Supports multi-GPU and multi-node inference,
including conversion and deployment of common large models - Support the
construction and transformation of new models - Support Triton inference
service framework - Supports multiple NVIDIA architectures: Volta,
Turing, Ampere, Hopper, Ada Lovelace - Quantization: FP16, FP8, INT8 &
INT4 Weight-Only, SmoothQuant, Groupwise (AWQ/GPTQ) - Supports FP8/INT8
KV Cache

------------------------------------------------------------------------

### ExLlamaV2

An inference library for running local LLMs on modern consumer GPUs.

**Key Features:** - New codebase and kernel implementation - Supports
the same 4-bit GPTQ model as V1 and the new EXL2 format - Supports
2--8-bit quantization, allowing mixing of quantization levels for
optimal bitrate - Integrates with Hugging Face and provides conversion
scripts

**Quantization Methods:** GPTQ, EXL2\
**References:**\
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)\
- [AWQ Paper](https://arxiv.org/abs/2306.00978)

------------------------------------------------------------------------

## Quantization and Inference

### vLLM

1.  Use `autoGPTQ` / `autoAWQ` to quantize the model
    -   Ref: [HuggingFace Quantization
        Docs](https://huggingface.co/docs/transformers/main_classes/quantization)
2.  Use vLLM to load and run inference

``` python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="./model/llama2_7b_awq_4b", quantization="awq")
outputs = llm.generate(prompts, sampling_params)
```

------------------------------------------------------------------------

### ExLlamaV2

1.  Use `exllamav2/convert.py` to quantize into EXL2 format
    -   Ref: [exllamav2 Convert
        Guide](https://github.com/turboderp/exllamav2/blob/master/doc/convert.md)

``` bash
python convert.py     -i /mnt/models/llama2-7b-fp16/     -o /mnt/temp/exl2/     -cf /mnt/models/llama2-7b-exl2/3.0bpw/     -b 3.0
```

2.  Load and run inference:

``` python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
import time

model_dir = "/home/ubuntu/xiaotong/projects/PII_check_LLM/model/exl2_8/"
config = ExLlamaV2Config(model_dir=model_dir)
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.05

prompt = "Our story begins in the Scottish town of Auchtermuchty, where once"
max_new_tokens = 150

generator.warmup()
start = time.time()
output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234)
end = time.time()

print(f"Generated in {end - start:.2f}s: {output}")
```

Ref: [ExLlamaV2 Inference
Example](https://github.com/turboderp/exllamav2/blob/master/examples/inference.py)

------------------------------------------------------------------------

### TensorRT-LLM

1.  Convert the model from HF checkpoint to TensorRT-LLM checkpoint
    format\
    Ref: [NVIDIA TensorRT-LLM LLaMA
    Example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)

``` bash
python convert_checkpoint.py     --model_dir ./tmp/llama/7B/     --output_dir ./tllm_checkpoint_1gpu_fp16_wq     --dtype float16     --use_weight_only     --weight_only_precision int8
```

2.  Build engine and run inference:

``` bash
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq              --output_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/              --gemm_plugin float16

python3 ../run.py     --max_output_len=50     --tokenizer_dir ./tmp/llama/7B/     --engine_dir=./tmp/llama/7B/trt_engines/weight_only/1-gpu/
```

TensorRT-LLM backend integrates with [Triton Inference
Server](https://github.com/triton-inference-server/tensorrtllm_backend).

------------------------------------------------------------------------

## Experiment Setup

**Task:** PII Filter\
**Environment:** - OS: Ubuntu 22.04\
- Instance: AWS x5.large\
- GPU: NVIDIA A10G ×1\
- Python: 3.10.12\
- CUDA: 12.2

**Input Example:**

    Original text: Your Amazon.com order  
    Masked text: Your Amazon.com order

------------------------------------------------------------------------

## Results

  ------------------------------------------------------------------------------
  Inference      Input    Output    Batch   Quantization   Speed         Time
  Library        Len      Len                              (tokens/s)    (s)
  -------------- -------- --------- ------- -------------- ------------- -------
  Transformers   180      24        1       INT8           10.4          2.3

  ExLlamaV2      180      24        1       EXL8           48            0.50

  ExLlamaV2      180      24        1       EXL5           62.7          0.38

  ExLlamaV2      180      24        6       EXL5           147.2         0.978

  vLLM           180      24        1       AWQ 4bit       75.9          0.316

  vLLM           180      24        1       GPTQ 4bit      69            0.35

  vLLM           180      24        1       FP8 KV Cache   32            0.769

  TensorRT-LLM   180      24        1       FP16           32            0.747

  TensorRT-LLM   180      24        1       INT8           62            0.387
                                            Weight-Only                  

  TensorRT-LLM   180      24        1       AWQ 4bit       96.7          0.248

  TensorRT-LLM   180      24        1       GPTQ 4bit      94.1          0.255
  ------------------------------------------------------------------------------

------------------------------------------------------------------------

## Conclusion

-   **ExLlamaV2** performs well on single-GPU setups, easy to install
    and deploy, but requires a CUDA compiler.\
-   **TensorRT-LLM** offers the best performance overall. Both TensorRT
    and vLLM support *in-flight batching* (called *Continuous Batching*
    in vLLM) and *Paged Attention*, making them ideal for
    high-throughput serving.\
-   For distributed inference:
    -   **TensorRT-LLM** supports *Tensor Parallelism* and *Pipeline
        Parallelism*
    -   **vLLM** supports only *Tensor Parallelism*\
        → Therefore, **TensorRT-LLM** is the better multi-GPU solution.

------------------------------------------------------------------------

**JIRA:** [FEEDS-6004: Accelerate the inference speed of LLM PII filter
model](https://yipitdata5.atlassian.net/browse/FEEDS-6004)
