from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "my-slurm"  # change this to match your cluster config name

## science_86k_high_quality
INPUT_FILE = "/lustre/fsw/portfolios/llmservice/users/zihanl/datasets/foundational_qa/s3_data/sftdedup/gpt_oss_120b/science/science_86k_high_quality.jsonl"
OUTPUT_FILE = "/lustre/fsw/portfolios/llmservice/users/zihanl/datasets/foundational_qa/s3_data/sftdedup/gpt_oss_120b/science/science_86k_high_quality"

NUM_SOLUTIONS_TO_GENERATE = 1
DEPENDENT_JOBS = 0
STARTING_SEED = 0
NUM_CHUNKS = 4

## prompt_config path
# nemo_skills/prompt/config/gpt-oss/...

generate(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=64000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/default "
        # we currently implement native Python code tool through text completions API
        # as we found alternative implementations to have issues.
        # We will switch to the official responses API when the support is added
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        # gpt-oss generates a lot of code, so need to set max_code_executions high!
        # you can also add ++server.code_execution.code_execution_timeout=120 to match
        # the setting in the official system prompt, but we found this to not impact
        # the accuracy, so keeping the default of 10 seconds
        # "++code_execution=false "
        # "++server.code_execution.max_code_executions=100 "
        # settings to enable high reasoning and Python built-in tool
        "++chat_template_kwargs.reasoning_effort=high "
        # "++chat_template_kwargs.builtin_tools=[python] "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-sdg-science",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    # can customize the number of GPUs used
    server_gpus=8,
    input_file=INPUT_FILE,
    # generations will be here. Needs to be a mounted folder
    output_dir=OUTPUT_FILE,
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launch a sandbox alongside the job that will keep track of
    # ipython sessions with stateful code execution
    with_sandbox=False,
    # num_chunks=N will parallelize the workload across X nodes
    # dependent_jobs=M will schedule this many dependent jobs on Slurm
    # (useful if your cluster has a fixed timeout per job)
    # set these according to your cluster configuration
    num_chunks=NUM_CHUNKS,
    dependent_jobs=DEPENDENT_JOBS,
    starting_seed=STARTING_SEED,
    num_random_seeds=NUM_SOLUTIONS_TO_GENERATE,
)

## commands
# cd /lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/Skills
# conda activate nemoskills

# git commit to repo
# python scripts/synthesize_science.py
