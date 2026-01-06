from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills/eval-results/zihan-sft-eval/"
gpus=8
server_nodes=1
i=0
model_name = "zihan-sft-29916"
model_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/megatron-lm/checkpoints/sft_gptoss_v2_1_32nodes_allpurpose_5e-5_32_262144/safetensors-checkpoint-29916"

## run commands
# cd /lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills
# conda activate nemoskills

# python scripts/eval_sft_ckpt.py


for benchmark in ["aime25"]:
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=131000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=generic/math_sft_notool "
        ),
        cluster=cluster,
        expname=f"{model_name}-no-python",
        model=model_path,
        server_type='vllm',
        server_gpus=8,
        num_chunks=1,
        benchmarks=f"{benchmark}:32",
        server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
        output_dir=output_dir + model_name + "/no-python",
    )

