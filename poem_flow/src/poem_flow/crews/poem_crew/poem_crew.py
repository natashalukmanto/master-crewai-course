import os

from crewai import Agent, Crew, Process, Task, LLM

from crewai.project import CrewBase, agent, crew, task

from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
)


@CrewBase
class PoemCrew():
	"""Poem Crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def poem_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['poem_writer'],
		)

	@task
	def write_poem(self) -> Task:
		return Task(
			config=self.tasks_config['write_poem'],
		)
	
	@crew
	def crew(self) -> Crew:
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			llm=gemini_llm
		)