import asyncio
import os
import json

from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_client =OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("LLM_API_KEY"), 
            base_url=os.getenv("LLM_API_BASE_URL"),
        )
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    """
        Core part: process query, calling MCP tool, produce final output.
    """
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema                
            }

        } for tool in response.tools]

        # Initial Claude API call
        response = self.llm_client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        while response.choices is not None and len(response.choices) > 0:
            choice = response.choices[0]
            response.choices = []

            if not hasattr(choice, "message"):
                break
            message = choice.message

            # Add assistant message
            messages.append(message)

            if message.content is not None:
                final_text.append(message.content)
            # Case that tool-calling is involved
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:

                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, json.loads(tool_args))
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    messages.append({
                        "role": "tool",
                        "content": [
                            {
                                # type is tool_result in Anthropic api
                                "type": "text",
                                "text":str( cont.text)
                            }
                            for cont in result.content
                        ],
                        "tool_call_id": tool_call.id,
                    })

                    response = self.llm_client.chat.completions.create(
                        model="qwen2.5-72b-instruct",
                        max_tokens=1000,
                        messages=messages,
                        tools=available_tools
                    )
        
        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())