import asyncio
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables (.env) BEFORE importing langsmith/langchain
# so LANGSMITH_API_KEY, LANGSMITH_TRACING, and LANGSMITH_PROJECT are active immediately.
load_dotenv()

import httpx
import langsmith
from langsmith import Langsmith

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import StructuredTool
from langchain_aws import ChatBedrockConverse

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --------------------------------------------------
# 0. LANGSMITH CONFIGURATION & CLIENT SETUP
# --------------------------------------------------

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "telecom-retention-copilot")

# Initialize official LangSmith client with retries and timeout options
ls_client = Langsmith(
    max_retries=2,  # Default retries for connection errors, 408, 409, 429, >=500
    timeout=httpx.Timeout(60.0, read=60.0, write=60.0, connect=20.0),
)


# --------------------------------------------------
# 1. MODEL
# --------------------------------------------------

model = ChatBedrockConverse(
    model_id="eu.amazon.nova-pro-v1:0",
    region_name="eu-north-1",
    temperature=0.0
)


# --------------------------------------------------
# 2. SPECIALIST PROMPTS
# --------------------------------------------------

RISK_PROMPT = """
You are the Risk Analyst of a Telecom Retention Copilot.

Your responsibility is to:
- Retrieve customer information using get_customer.
- Predict churn using predict_churn.
- Report churn probability, risk percentage and risk category.
- Use actual tool results only.
- Never invent customer information or prediction results.

Do not search retention policies or create cases.
Return concise, structured findings to the supervisor.
"""

POLICY_PROMPT = """
You are the Policy Expert specialist for the Telecom Customer
Retention Copilot.

Your responsibility is to search retention policy documents,
discount structures, eligibility rules, and guidelines using
search_policies.

Instructions:
- Use search_policies for policy-related questions.
- Identify all relevant matching policies and eligible offers.
- Do not return only the highest discount tier when multiple
  offers are eligible.
- Explain discount amounts, durations, eligibility conditions,
  and relevant restrictions.
- Distinguish offer eligibility from approval authority.
- Do not invent policy rules or discount amounts.
- If the retrieved information is insufficient, say so.
- Keep your response concise.

Do not predict churn or create cases.
Return concise policy findings to the supervisor.
"""

ACTION_PROMPT = """
You are the Action Drafter of a Telecom Retention Copilot.

Your responsibility is to:
- Prepare customer retention recommendations.
- Combine the customer risk information and applicable policies.
- Suggest an appropriate retention offer.
- Explain why the offer is suitable.
- Draft a case when the user explicitly requests case creation.
- When the user explicitly requests creating/opening/submitting a case, call the create_case tool using the relevant customer, policy/offer, and reason details from the conversation.
- Do NOT just write a text draft when asked to create a case—invoke create_case so human approval can run.

Never invent customer information, churn scores or policy rules.

You have access to create_case, but you must only call it
when the user explicitly requests case creation.

The create_case tool requires human approval before execution.
Never claim that a case was created unless the tool confirms it.

Return concise results to the supervisor.
"""


# --------------------------------------------------
# 3. SUPERVISOR PROMPT
# --------------------------------------------------

SUPERVISOR_PROMPT = """
You are the supervisor of a Telecom Retention Copilot.

You manage three specialist agents:

1. risk_analyst:
    Retrieves customer details and predicts churn.

2. policy_expert:
    Searches retention policies and determines eligibility.

3. action_drafter:
    Prepares retention recommendations and creates cases
    only when explicitly requested by the user.

Delegation rules:

- For customer details or churn-related requests, use risk_analyst.
- For retention policy questions, use policy_expert.
- For retention recommendations, use the relevant specialists
  and then action_drafter.
- For ANY request to create, open, log, or submit a case -> gather needed risk/policy info first, and ALWAYS transfer to action_drafter before giving your final response. Never answer a case-creation request yourself without calling action_drafter.
- Do not invoke unnecessary agents.
- Use actual specialist results. Never invent information.
- Do not ask specialists to repeat work already completed.
- Present only the consolidated final answer.
- Never claim that a case was created without confirmation.
"""


def get_approval(decision: str) -> bool:
    """Normalize user input and return True if approval is granted."""
    accepted_values = {"yes", "y", "1", "true"}
    return decision.strip().lower() in accepted_values


def extract_text_response(content) -> str:
    """Extract clean text from Nova Pro structured content blocks."""
    if isinstance(content, list):
        text_parts = [
            block["text"].strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        ]
        return "\n".join(text_parts)
    return str(content).strip()


# --------------------------------------------------
# 4. HUMAN-IN-THE-LOOP CREATE_CASE WRAPPER
# --------------------------------------------------

def protect_create_case(original_tool):

    async def guarded_create_case(**kwargs):
        # Pause execution before calling the original MCP tool.
        approval = interrupt({
            "action": "create_case",
            "message": "Human approval required before creating a case.",
            "arguments": kwargs
        })

        # Execute only if the human explicitly approves.
        if not isinstance(approval, dict) or not approval.get("approved"):
            return "Case creation cancelled. Human approval was not granted."

        # Only now execute the actual MCP tool.
        result = await original_tool.ainvoke(kwargs)
        return result

    return StructuredTool(
        name=original_tool.name,
        description=(
            original_tool.description
            + " Human approval is required before execution."
        ),
        args_schema=original_tool.args_schema,
        coroutine=guarded_create_case
    )


# --------------------------------------------------
# 5. MAIN APPLICATION
# --------------------------------------------------

async def main():

    server_path = Path(__file__).resolve().parent / "mcp_server.py"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            try:
                await session.initialize()
            except Exception as e:
                print(f"MCP Session Initialization Error: {e}")
                return

            # Load all existing MCP tools.
            mcp_tools = await load_mcp_tools(session)

            print("MCP server connected.")

            # Create a dictionary for easy tool selection.
            tools = {
                tool.name: tool
                for tool in mcp_tools
            }

            # --------------------------------------------------
            # 6. CREATE HUMAN-APPROVED CASE TOOL
            # --------------------------------------------------

            protected_case_tool = protect_create_case(
                tools["create_case"]
            )

            # --------------------------------------------------
            # 7. CREATE THREE SPECIALIST AGENTS
            # --------------------------------------------------

            risk_agent = create_agent(
                model=model,
                tools=[
                    tools["get_customer"],
                    tools["predict_churn"]
                ],
                name="risk_analyst",
                system_prompt=RISK_PROMPT
            )

            policy_agent = create_agent(
                model=model,
                tools=[
                    tools["search_policies"]
                ],
                name="policy_expert",
                system_prompt=POLICY_PROMPT
            )

            action_agent = create_agent(
                model=model,
                tools=[
                    protected_case_tool
                ],
                name="action_drafter",
                system_prompt=ACTION_PROMPT
            )

            print("Three specialist agents created.")

            # --------------------------------------------------
            # 8. CREATE SUPERVISOR WORKFLOW
            # --------------------------------------------------

            workflow = create_supervisor(
                agents=[
                    risk_agent,
                    policy_agent,
                    action_agent
                ],
                model=model,
                prompt=SUPERVISOR_PROMPT,
                output_mode="last_message"
            )

            # --------------------------------------------------
            # 9. SQLITE CHECKPOINTING
            # --------------------------------------------------

            async with AsyncSqliteSaver.from_conn_string(
                "retention_checkpoints.sqlite"
            ) as checkpointer:

                app = workflow.compile(
                    checkpointer=checkpointer
                )

                print("\nTelecom Retention Copilot is ready (Tracing Enabled).")

                thread_id = input(
                    "\nEnter existing Thread ID "
                    "or press Enter for a new conversation: "
                ).strip()

                if not thread_id:
                    thread_id = str(uuid.uuid4())

                print(f"\nThread ID: {thread_id}")
                print("Save this ID to continue this conversation later.")

                config = {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }

                # --------------------------------------------------
                # 10. INTERACTIVE CONVERSATION & ERROR HANDLING
                # --------------------------------------------------

                while True:

                    user_input = input("\nYou: ").strip()

                    if user_input.lower() in ["exit", "quit"]:
                        print("Conversation ended.")
                        break

                    if not user_input:
                        continue

                    try:
                        # Run the supervisor with automatic LangSmith tracing
                        result = await app.ainvoke(
                            {
                                "messages": [
                                    ("user", user_input)
                                ]
                            },
                            config=config
                        )

                        # Handle human approval interrupt if triggered
                        if result.get("__interrupt__"):
                            for pending in result["__interrupt__"]:
                                print("\nHUMAN APPROVAL REQUIRED")
                                print(pending.value)

                                decision = input(
                                    "\nApprove case creation? (yes/no): "
                                )

                                approved = get_approval(decision)

                                result = await app.ainvoke(
                                    Command(
                                        resume={
                                            "approved": approved
                                        }
                                    ),
                                    config=config
                                )

                        final_message = result["messages"][-1]

                        print("\nCopilot:")
                        print(extract_text_response(final_message.content))

                    # Official LangSmith SDK Exception Hierarchy Handling
                    except langsmith.APIConnectionError as e:
                        print("\nThe server could not be reached.")
                        print(e.__cause__)
                    except langsmith.RateLimitError:
                        print("\nA 429 status code was received; we should back off a bit.")
                    except langsmith.APITimeoutError:
                        print("\nThe request timed out.")
                    except langsmith.APIStatusError as e:
                        print(f"\nAnother non-200-range status code was received: {e.status_code}")
                        print(e.response)
                    except langsmith.APIError as e:
                        print(f"\nA general LangSmith API error occurred: {e}")
                    except Exception as e:
                        print(f"\nExecution Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())