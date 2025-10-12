from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "my-slurm"  # change this to match your cluster config name

# with python
eval(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=120000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        # we currently implement native Python code tool through text completions API
        # as we found alternative implementations to have issues.
        # We will switch to the official responses API when the support is added
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        # gpt-oss generates a lot of code, so need to set max_code_executions high!
        # you can also add ++server.code_execution.code_execution_timeout=120 to match
        # the setting in the official system prompt, but we found this to not impact
        # the accuracy, so keeping the default of 10 seconds
        "++code_execution=true "
        "++server.code_execution.max_code_executions=100 "
        # settings to enable high reasoning and Python built-in tool
        "++chat_template_kwargs.reasoning_effort=high "
        "++chat_template_kwargs.builtin_tools=[python] "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-eval-with-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    # can customize the number of GPUs used
    server_gpus=8,
    benchmarks="aime24:16,aime25:16",
    # generations and metrics will be here. Needs to be a mounted folder
    output_dir="/lustre/fsw/portfolios/llmservice/users/dongfuj/Workspace/Skills/gpt-oss-eval/with-python",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launch a sandbox alongside the job that will keep track of
    # ipython sessions with stateful code execution
    with_sandbox=True,
    # launching all benchmarks / samples on the same node
    # for bigger benchmarks, you can adjust this accordingly
    # num_jobs is the number of copies of the server you can use to parallelize evaluation
    # the total amount of GPUs used is server_gpus x server_nodes x num_jobs
    num_jobs=1,
)


# # without python
# eval(
#     ctx=wrap_arguments(
#         # not specifying tokens_to_generate here, by default uses all available context

#         # recommended inference settings including prompt config
#         "++inference.temperature=1.0 "
#         "++inference.top_p=1.0 "
#         "++prompt_config=gpt-oss/math "
#         # setting reasoning effort through vllm arguments as we are using chat completions api here
#         "++inference.extra_body.reasoning_effort=high "
#     ),
#     cluster=cluster,
#     # optional parameter here, but useful when chaining multiple jobs together in pipelines
#     expname="gpt-oss-eval-no-python",
#     model="openai/gpt-oss-120b",
#     server_type='vllm',
#     # can customize the number of GPUs used
#     server_gpus=8,
#     benchmarks="aime24:16,aime25:16",
#     # generations and metrics will be here. Needs to be a mounted folder
#     output_dir="/workspace/gpt-oss-eval/no-python",
#     # any vllm arguments can be used here
#     server_args="--async-scheduling",
#     # launching all benchmarks / samples on the same node in parallel
#     # for bigger benchmarks, you can adjust this accordingly
#     # num_jobs is the number of copies of the server you can use to parallelize evaluation
#     # the total amount of GPUs used is server_gpus x server_nodes x num_jobs
#     num_jobs=1,
# )