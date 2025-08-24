from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from deepagents import create_deep_agent


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


@tool
def run_python_repl(code: str) -> str:
    """Run Python code in a REPL environment. Only the print statements will be returned."""
    repl = PythonREPL()
    result = repl.run(code)
    return result


sub_agent_prompt = """You are a specialized researcher. Your job is to find information about a given codebase to answer the user's questions.
You have access to 'run_python_repl' tool that can help you navigate the codebase or reading files, directories, etc. Remember, the tool only returns the output of print statements, so make sure to add proper imports and print statements in the code snippets.
Conduct thorough research and then reply to the user with a concise answer to their question.
Only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message."""

code_walk_agent = {
    "name": "code-walk-agent",
    "description": "Used to find information about a codebase. Only give this worker agent one topic at a time. Do not pass multiple sub questions to this worker. Instead, you should break down a large topic into the necessary components, and then call multiple worker agents in parallel, one for each sub question.",
    "prompt": sub_agent_prompt,
    "tools": ["run_python_repl"],  # tool name as a string
}


readme_instructions = """You are an expert in software development and documentation.

You will be given a repo link to begin with and your task is to create a comprehensive README file for a software project.

The first thing you should do is to write the original user message to `question.txt` so you have a record of it.

Use the code-walk-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final README file, write it to `final_report.md`

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).


This README should include the following sections:

1. Overview
2. Description
3. Installation Instructions
4. Usage
5. Contributing
6. License

Make sure to provide clear and concise information in each section. Use code blocks where necessary to illustrate usage examples or installation steps.

(Optional):
You are allowed to add comments and placeholders for media items or information that can make the README more engaging and informative.
"""

# Create the agent
agent = create_deep_agent(
    model=llm,
    tools=[run_python_repl],
    instructions=readme_instructions,
    subagents=[code_walk_agent],
).with_config({"recursion_limit": 10})
