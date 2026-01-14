from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/results/eval-longcontext/"
gpus=8
server_nodes=1
i=0

# model_name = "nano-v3-30b-a3b-sft"
# model_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/checkpoint/nvidia-nemotron-3-nano-30b-a3b-sft"

model_name = "nano-v3-30b-a3b"
model_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/checkpoint/nvidia-nemotron-3-nano-30b-a3b-bf16"


benchmark = "aalcr"

eval(
    ctx=wrap_arguments(
        "++inference.tokens_to_generate=131000 "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=generic/default "
    ),
    cluster=cluster,
    data_dir="/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/nemo_skills/dataset",
    expname=f"{model_name}-aalcr",
    model=model_path,
    server_type='vllm',
    server_gpus=8,
    num_chunks=1,
    benchmarks=f"{benchmark}:16",
    server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
    output_dir=output_dir + model_name,
    judge_model='Qwen/Qwen3-235B-A22B-Instruct-2507',
    judge_server_type='sglang',
    judge_server_gpus=8,
)


## run commands preparation
# cd /lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills
# conda activate nemoskills

## download data
# ns prepare_data --data_dir=/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/nemo_skills/dataset --cluster=slurm aalcr

## run script
# git commit to repo or
# export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1
# python scripts/eval_sft_ckpt_aalcr.py


## check final acc
# ns summarize_results --cluster=slurm /lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/results/eval-longcontext/nano-v3-30b-a3b-sft/eval-results/aalcr
# ns summarize_results --cluster=slurm OUTPUT_DIR
