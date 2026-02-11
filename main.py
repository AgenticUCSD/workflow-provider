from utils.model import model
from utils.tools import curl
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware import SkillsMiddleware, FilesystemMiddleware

import uuid
import os

# we're going to use a checkpointer since we're giving the agent edit permissions
checkpointer = MemorySaver()

# we need an absolute path, but not a windows absolute path
root_dir = os.path.abspath("./agent_data")
root_dir = root_dir.replace("\\", "/")
backend = FilesystemBackend(root_dir=root_dir)
skills = [f"{root_dir}/skills/"]


my__middleware = [
    TodoListMiddleware(),
    SkillsMiddleware(backend=backend, sources=skills),
    FilesystemMiddleware(backend=backend),
]

# agent with our custom model and skill middleware
agent = create_agent(
    model = model,
    tools=[curl],
    checkpointer=checkpointer,
    middleware=my__middleware,
    system_prompt=(
        "**CRITICAL INSTRUCTION - SKILL USAGE PROTOCOL:**\n\n"
        "BEFORE responding to ANY user request, you MUST:\n"
        "1. Check the Available Skills list below\n"
        "2. Compare the user's request against each skill's description and trigger phrases\n"
        "3. If ANY skill even barely matches, you MUST READ that skill's SKILL.md and follow it\n"
        "4. NEVER provide a direct answer if a matching skill exists\n\n"
        "**Skill Matching Rules:**\n"
        "- User says words from a skill description → USE that skill\n\n"
        "Skills contain specialized workflows you MUST follow. Do NOT improvise "
        "if a skill exists for the task. Read and execute the skill's instructions.\n\n"
        "**WARNING: DO NOT USE THE TASK TOOL AND DO NOT USE SUBAGENTS.**\n\n"
    ),
)

# Configuration for this conversation thread
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}


with open("prompts/download_video_input.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

# Initial message
result = agent.invoke(  
    {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    },
    config
)

# Print the conversation
for message in result["messages"]:
    if hasattr(message, 'pretty_print'):
        message.pretty_print()
    else:
        print(f"{message.type}: {message.content}")

# Continue conversation loop
while True:
    user_input = input("\nYour response (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config  # Same config preserves conversation state
    )
    
    # Print latest response
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")