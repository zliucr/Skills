"""
Custom generation task for JSON tool call format:
<tool_call>
{"name": "python", "arguments": {"code": "..."}}
</tool_call>
<tool_response>
output
</tool_response>

This module provides a custom CodeExecutionWrapper that parses JSON to extract code.
"""

import json
import re
import logging
import sys
from typing import Dict, Any

from nemo_skills.inference.model.code_execution import CodeExecutionWrapper, CodeExecutionConfig
from nemo_skills.inference.model.base import BaseModel
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.generate import GenerationTask, GenerateSolutionsConfig
from nemo_skills.inference.model import get_model
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_from_json_tool_call(text: str, code_begin: str, code_end: str) -> str:
    """Extract Python code from JSON tool call format.

    Input format:
    <tool_call>
    {"name": "python", "arguments": {"code": "print('hello')"}}
    </tool_call>

    Returns the code string from arguments.code
    """
    # Find content between tags
    pattern = re.escape(code_begin) + r'(.*?)' + re.escape(code_end)
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return ""

    json_str = match.group(1).strip()

    try:
        parsed = json.loads(json_str)
        code = parsed.get('arguments', {}).get('code', '')
        return code
    except json.JSONDecodeError as e:
        LOG.warning(f"Failed to parse JSON tool call: {e}")
        LOG.warning(f"JSON string was: {json_str[:200]}...")
        return ""


def format_tool_response(output: str) -> str:
    """Format code execution output as <tool_response>."""
    return f"<tool_response>\n{output}\n</tool_response>"


class JsonToolCallCodeExecutionWrapper(CodeExecutionWrapper):
    """CodeExecutionWrapper that handles JSON tool call format."""

    async def execute_generated_code(self, input_prompt, code_begin, code_end, output, session_id):
        """Override to extract code from JSON format."""
        import time

        code_execution_time_start = time.time()

        # Extract code from JSON tool call format
        header = "\n".join(self.config.code_execution_headers)
        code_block = extract_code_from_json_tool_call(output, code_begin, code_end)

        if not code_block:
            LOG.warning("No code extracted from tool call")
            return code_execution_time_start, {
                "process_status": "error",
                "stdout": "",
                "stderr": "Failed to extract code from tool call"
            }, session_id

        extracted_code = f"{header}{code_block}"

        execution_dict, session_id = await self.sandbox.execute_code(
            generated_code=extracted_code,
            language=self.config.code_execution_language,
            timeout=self.config.code_execution_timeout,
            max_output_characters=self.config.max_code_output_characters,
            session_id=session_id,
            traceback_verbosity=self.config.sandbox_traceback_verbosity,
        )

        return code_execution_time_start, execution_dict, session_id


def get_json_tool_code_execution_model(server_type, tokenizer=None, code_execution=None, sandbox=None, **kwargs):
    """Get a model wrapped with JSON tool call code execution."""
    model = get_model(server_type=server_type, tokenizer=tokenizer, **kwargs)
    if code_execution is None:
        code_execution = {}
    code_execution_config = CodeExecutionConfig(**code_execution)
    return JsonToolCallCodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)


class JsonToolCallGenerationTask(GenerationTask):
    """Generation task that uses JSON tool call format for code execution."""

    def setup_llm(self):
        from nemo_skills.code_execution.sandbox import get_sandbox

        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None

        if self.cfg.code_execution:
            # Use our custom JSON tool call wrapper
            # Note: code_execution config is already in self.cfg.server
            llm = get_json_tool_code_execution_model(
                **self.cfg.server,
                tokenizer=self.tokenizer,
                sandbox=self.sandbox,
            )
        else:
            from nemo_skills.inference.model import get_model
            llm = get_model(**self.cfg.server, tokenizer=self.tokenizer)

        return llm


# This is what nemo_skills pipeline looks for
GENERATION_TASK_CLASS = JsonToolCallGenerationTask


# Entry point for running as a script
import hydra
from nemo_skills.inference.generate import GenerateSolutionsConfig, get_help_message, server_params, sandbox_params
from nemo_skills.utils import setup_logging


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = JsonToolCallGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()