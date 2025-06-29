import gradio as gr
from crewai import Agent, Task, LLM, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

# Define your LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

# Define your tool
search_tool = SerperDevTool(n=2)

# Define your agents
def generate_report(topic):
    Senior_Research_Analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze and synthesize information on the {topic}",
        backstory="You are a senior research analyst with expertise in web research and analysis."
                  "You excel at finding relevant information, analyzing and synthesizing"
                  "information from across the internet using search tools. You are skilled at"
                  "distinguishing reliable sources from unreliable ones."
                  "Fact checking, cross-referencing information, and identifying key patterns and insights."
                  "You provide well organized briefs with proper citations."
                  "Your analysis includes both raw data and interpreted insights,"
                  "making complex information accessible and actionable.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )

    content_writer = Agent(
        role="Content Writer",
        goal="Create a comprehensive and engaging content piece based on the research findings provided by the Senior Research Analyst.",
        backstory="You are a skilled content writer specializing in creating high-quality, engaging"
                  "accessible content from technical research."
                  "You work closely with the Senior Research Analyst to ensure that the content is"
                  "balanced between informative and entertaining writing,"
                  "while ensuring all facts and citations from the research are properly incorporated."
                  "You have a talent for making complex topics approachable without oversimplifying them.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Define tasks
    research_task = Task(
        name="Research Task",
        description=f"Conduct research on the {topic} and provide a comprehensive report."
                    "Evaluate the sources and fact check all information."
                    "Organize the findings in a structured format with proper citations.",
        expected_output=f"A well-organized research report containing:"
                        "Executive summary of key findings,"
                        "Comprehensive analysis of current trends and developments,"
                        "List of verified facts and statistics,"
                        "All citations and links to original sources,"
                        "Clear categorization of main themes and patterns.",
        agent=Senior_Research_Analyst,
        verbose=True,
    )

    writing_task = Task(
        name="Content Writing Task",
        description="Create a comprehensive and engaging content piece based on the research findings provided by the Senior Research Analyst.",
        expected_output="A well-structured content piece that includes:"
                        "An engaging introduction to the topic,"
                        "Key insights and findings from the research,"
                        "Clear explanations of complex concepts,"
                        "Proper citations and references,"
                        "A conclusion that summarizes the main points and provides actionable insights.",
        agent=content_writer,
        verbose=True,
    )

    # Define crew
    crew = Crew(
        agents=[Senior_Research_Analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    result = crew.kickoff()
    return result


# Gradio interface
interface = gr.Interface(
    fn=generate_report,
    inputs=gr.Textbox(label="Enter a topic"),
    outputs=gr.Textbox(label="Generated Content"),
    title="AI Research & Content Generator",
    description="Enter a topic and generate a research report + content piece using CrewAI!"
)

if __name__ == "__main__":
    interface.launch()
