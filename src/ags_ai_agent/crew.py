from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from typing import List
from typing import ClassVar
from src.ags_ai_agent.tools.vector_db import ingest_documents
from src.ags_ai_agent.tools.custom_tool import MyCustomTool
import time
from src import state
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Create a PDF knowledge source
# pdf_source = PDFKnowledgeSource(
#     file_paths=["AlmondButterSpec.pdf", "BlackSesameButterSpec.pdf", "PistachioButterSpec.pdf", "FAQ_B2B.pdf"]
# )

# ingest_documents("knowledge")

milvus_tool = MyCustomTool()

slm = LLM(model="gpt-4o-mini")
llm = LLM(model="gpt-4o")
# class Input(BaseTool):
#     name: ClassVar[str] = "get_input"
#     description: str = "Get information from the user. Ensure all necessary info is collected"

#     def _run(self, prompt: str) -> str:
#         print(prompt)
#         user_input = input("> ")
#         return user_input

class Input(BaseTool):
    name: ClassVar[str] = "get_input"
    description: str = "Get info from user via GUI chatbox"

    def _run(self, prompt: str) -> str:
        state.last_prompt = prompt
        state.awaiting_input = True
        while state.last_response == "":
            time.sleep(0.1)
        response = state.last_response
        state.last_response = ""
        state.awaiting_input = False
        return response

@CrewBase
class AgsAiAgent():
    """AgsAiAgent crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents: List[BaseAgent]
    tasks: List[Task]


    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def receiver(self) -> Agent:
        
        return Agent(
            config=self.agents_config['receiver'],
            tools=[Input(), milvus_tool],
            # knowledge_sources=[pdf_source],
            verbose=True,
            llm=llm,
            initialize_state=lambda: {
            "requirements_collected": [],
            "pending_requirements": [
                "Product Specifications", "Certifications", "Shelf Life", "INCOTERM",
                "Order Volume", "Packaging Requirements", "Payment Terms", "Target Price"
            ]
        }
        )
    
    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config['manager'], # type: ignore[index]
            # knowledge_sources=[pdf_source],
            tools=[milvus_tool],
            verbose=True,
            llm=slm
        )

    @agent
    def analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['analyzer'], # type: ignore[index]
            # knowledge_sources=[pdf_source],
            tools=[milvus_tool],
            verbose=True,
            llm=slm
        )
    
    @agent
    def compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['compiler'], # type: ignore[index]
            # knowledge_sources=[pdf_source],
            tools=[milvus_tool],
            verbose=True,
            llm=slm
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def get_input(self) -> Task:
        return Task(
            config=self.tasks_config['collect_customer_requirements_task'], # type: ignore[index]
            agent=self.receiver(), # type: ignore[index]
            memory=False,
            cache=False,
        )

    @task
    def manage_task(self) -> Task:
        return Task(
            config=self.tasks_config['sales_planner_task'], # type: ignore[index]
            context=[self.get_input()],
            agent = self.manager(), # type: ignore[index]
            memory=False,
            cache=False,
        )

    @task
    def analyze_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_match_task'], # type: ignore[index]
            context=[self.get_input(), self.manage_task()],
            agent = self.analyzer(), # type: ignore[index]
            memory=False,
            cache=False,
        )
    
    @task
    def compile_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_proposal_task'], # type: ignore[index]
            context=[self.manage_task(), self.analyze_task()], # type: ignore[index]
            output_file='report.md',
            agent=self.compiler(),
            memory=False,
            cache=False,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the LeadingAi crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            # knowledge_sources=[pdf_source],
            tools=[milvus_tool],
            memory=False,
            cache=False,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
