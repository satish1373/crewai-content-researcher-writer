from crewai import Agent, Task, LLM, Crew
from crewai_tools import SerperDevTool

from dotenv import load_dotenv
load_dotenv()


topic = "Artificial Intelligence in Healthcare"

# Agent 1 - researcher==> Websearch (Serper API)
#Agent2 - Content Creator (Summarization)

# Initialize the LLM and provide the model name and temperature

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

#Tool for web search
search_tool = SerperDevTool(n=2)

# Agent 1 - Create the first agent for web search
Senior_Research_Analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Research, analyze and synthezise information on the {topic}",
    backstory = "You are a senior research analyst with expertise in web research and analysis."
                "You excel at finding relevant information, analyzing and synthesizing"
                "information from across the internet using search tools. You are skileld at"
                "distingusing reliable sources from unreliable ones."
                "fact checking, cross-referencing information, and identifying key patterns and insights."
                "you provide well organized briefs with proper citations."
                "and source verification. Your analysis includes both raw data and interpreted insights,"
                "making complex infromation accessible and actionable.",
    verbose = True,
    allow_delegation=False,
    tools = [search_tool],
    llm = llm,
)

#agent 2 - Content creation

content_writer = Agent(
    role = "Content Writer",
    goal = "Create a comprehensive and engaging content piece based on the research findings provided by the Senior Research Analyst.",
    backstory = "You are a skilled content writer specializing in creating high-quality, engaging"
                "accessible content from technical research."
                "you work closely with the Senior Research Analyst to ensure that the content is"
                "balance between ifnroamtive and entertaining writing"
                "while ensuring all the the facts and citations from the research"
                "are properly incorporated. You have a talent for making complex topics"
                "approachable without oversimplifying them.",
    verbose = True,
    allow_delegation=False,
    llm = llm,
)

# Define the Task
# Research Task
research_task = Task(
    name="Research Task",
    description=f"Conduct research on the {topic} and provide a comprehensive report."
                "Evaluate the sources and fact check all information."
                "organize the findings in a structured format with proper citations.",
    expected_output=f"A well-organized research report containing:"
                    "executive summary if key findings"
                    "Comprehensive analysis of current trends and developments"
                    "list of verified fact and statistics"
                    "All citations and linls to original sources"
                    "clear categorization of main theme and patterns"
                    "Please format with clear sections and bullet points for readability.",
    agent=Senior_Research_Analyst,
    verbose=True,
)

#Content Writer Task
Writing_task = Task(
    name="Content Writing Task",
    description="Create a comprehensive and engaging content piece based on the research findings provided by the Senior Research Analyst.",
    expected_output="A well-structured content piece that includes:"
                    "An engaging introduction to the topic"
                    "Key insights and findings from the research"
                    "Clear explanations of complex concepts"
                    "Proper citations and references to the research sources"
                    "A conclusion that summarizes the main points and provides actionable insights.",
    agent=content_writer,
    verbose=True,
)


# Create the Crew
crew = Crew(
    agents=[Senior_Research_Analyst, content_writer],
    tasks=[research_task, Writing_task],
    verbose=True,
)

result = crew.kickoff()

