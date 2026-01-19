from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/results/eval-multichallenge/"
gpus=8
server_nodes=1
i=0

model_name = "nano-v3-30b-a3b"
model_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/checkpoint/nvidia-nemotron-3-nano-30b-a3b-bf16"

benchmark = "multi_challenge"

eval(
    ctx=wrap_arguments(
        "++inference.tokens_to_generate=131000 "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=generic/default_v2 "
    ),
    cluster=cluster,
    expname=f"{model_name}",
    model=model_path,
    server_type='vllm',
    server_gpus=8,
    num_chunks=1,
    benchmarks=f"{benchmark}:1",
    server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
    output_dir=output_dir + model_name,
)

## run commands
# cd /lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills
# conda activate nemoskills

## run script
# git commit to repo or
# export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

# python scripts/eval_sft_ckpt_math.py
