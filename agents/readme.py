from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama


# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm = ChatOllama(model="gpt-oss")


@tool
def run_python_repl(code: str) -> str:
    """Run Python code in a REPL environment. ALWAYS add proper imports and print statements to your code."""
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

You will be given a repo link or a directory path to begin with and your task is to create a comprehensive README file for a software project.

The first thing you should do is to write the user message/task to `question.txt` so you have a record of it.

Use the code-walk-agent to conduct deep research. It will respond to your request with a detailed answer. The agent works best when given one topic at a time.

When you think you enough information to write a final README file, write it to `final_report.md`

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

This README should have all the relevant sections including small snippets highlighting key features and usage examples.

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
).with_config({"recursion_limit": 1000})
