<h1 align="center">Hi there 👋, I'm Natasha!</h1>

<p align="center">
  A CS Student @ UC Berkeley studying crewAI!  
</p>

---

## About This Repo

In this repository, you will find what I learned about crewAI.
- Credit to [Tyler AI](https://youtu.be/ONKOXwucLvE?si=QwINdQNPEFPMq9wf) on YouTube

---

## Languages & Tools

- [crewAI](https://crewai.com)
- [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
- [![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](#)
- [![Anaconda](https://img.shields.io/badge/Anaconda-44A833?logo=anaconda&logoColor=fff)](#)

---

## Let's Connect

<p align="left">
<a href="https://www.linkedin.com/in/natasha-lukmanto" target="blank"><img align="center" src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin" /></a>
</p>

---

<p align="center">Thanks for visiting my repo!</p>

---

> **Note:** The following content below is authored by [crewAI](https://crewai.com). I did not contribute to its content.

---

# AiLatestDevelopment Crew

Welcome to the AiLatestDevelopment Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/ai_latest_development/config/agents.yaml` to define your agents
- Modify `src/ai_latest_development/config/tasks.yaml` to define your tasks
- Modify `src/ai_latest_development/crew.py` to add your own logic, tools and specific args
- Modify `src/ai_latest_development/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the ai-latest-development Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The ai-latest-development Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the AiLatestDevelopment Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
