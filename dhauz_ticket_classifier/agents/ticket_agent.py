"""LangChain agent wrapper (keeps the original agent code as optional)"""
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.tools import StructuredTool


class TicketAgent:
    def __init__(self, rag_classifier=None, llm=None):
        self.rag = rag_classifier
        self.llm = llm
        self.agent = None
        self.agent_executor = None

    def build_tool(self):
        if self.rag is None:
            raise ValueError("rag_classifier is required to build tool")

        tool = StructuredTool.from_function(
            func=self.rag.classify_batch,
            name="IT_Ticket_Classifier",
            description="Classify an IT support ticket into predefined categories"
        )
        return [tool]

    def create(self, system_prompt: str = None):
        tools = self.build_tool()
        system_prompt = system_prompt or "You are an AI assistant specialized in IT support ticket classification."
        if self.llm is None:
            raise ValueError("llm object required to create agent")
        self.agent = create_agent(model=self.llm, tools=tools, system_prompt=system_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
