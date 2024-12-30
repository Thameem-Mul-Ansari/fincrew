from flask import Flask, request
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
from tools import tool
from dotenv import load_dotenv
import os
from textwrap import dedent

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load API key
api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please set GEMINI_API_KEY environment variable.")

# Create LLM instance
llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.5,
    max_tokens=1024,
    top_p=0.9,
    stop=None,
    stream=False,
    api_key=api_key
)

# Define agents
financial_analyst = Agent(
    role="The Best Financial Analyst",
    goal="""Impress all customers with your financial data 
            and market trends analysis.""",
    backstory="""The most seasoned financial analyst with 
            lots of expertise in stock market analysis and investment
            strategies. Now working for an important customer.""",
    llm=llm,
    verbose=True
)

research_analyst = Agent(
    role="Staff Research Analyst",
    goal="""Be the best at gathering and interpreting data,
            and amaze your customers with it.""",
    backstory="""Known as the BEST research analyst, you're
            skilled in sifting through news, company announcements, 
            and market sentiments. Now you're working for an 
            important customer.""",
    llm=llm,
    verbose=True
)

investment_advisor = Agent(
    role="Private Investment Advisor",
    goal="""Impress your customers with full analyses of stocks
            and complete investment recommendations.""",
    backstory="""You're the most experienced investment advisor
            combining various analytical insights to formulate
            strategic investment advice. Now working for an 
            important customer.""",
    llm=llm,
    verbose=True
)


@app.route('/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json()
        stock_symbol = data.get('company')

        if not stock_symbol:
            return "Error: Stock symbol is required", 400

        # Define tasks dynamically with the user-provided company name
        research_task = Task(
            description=dedent(
                f"""
                Collect and summarize recent news articles, press
                releases, and market analyses related to the stock and
                its industry.
                Pay special attention to any significant events, market
                sentiments, and analysts' opinions. Also include upcoming 
                events like earnings and others.
                
                Your final answer MUST be a report that includes a
                comprehensive summary of the latest news, any notable
                shifts in market sentiment, and potential impacts on 
                the stock.
                Also make sure to return the stock ticker.
                Make sure to use the most recent data as possible.
                
                Selected company by the customer: {stock_symbol}
                """
            ),
            expected_output="A comprehensive summary report",
            tools=[tool],
            agent=research_analyst
        )

        financial_analysis_task = Task(
            description=dedent(
                f"""
                Conduct a thorough analysis of the stock's financial
                health and market performance. 
                This includes examining key financial metrics such as
                P/E ratio, EPS growth, revenue trends, and 
                debt-to-equity ratio. 
                Also, analyze the stock's performance in comparison 
                to its industry peers and overall market trends.

                Your final report MUST expand on the summary provided
                but now including a clear assessment of the stock's
                financial standing, its strengths and weaknesses, 
                and how it fares against its competitors in the current
                market scenario.

                Make sure to use the most recent data possible.
                """
            ),
            expected_output="A detailed financial analysis",
            tools=[tool],
            agent=financial_analyst
        )

        filings_analysis_task = Task(
            description=dedent(
                f"""
                Analyze the latest 10-Q and 10-K filings from EDGAR for  
                the stock in question.   
                Focus on key sections like Management's Discussion and  
                Analysis, financial statements, insider trading activity,   
                and any disclosed risks.  
                Extract relevant data and insights that could influence  
                the stock's future performance.  
                Your final answer must be an expanded report that now  
                also highlights significant findings from these filings,  
                including any red flags or positive indicators for  
                your customer.
                """
            ),
            expected_output="Insights from filings analysis",
            tools=[tool],
            agent=research_analyst,
        )

        investment_advice_task = Task(
            description=dedent(
                f"""
                Review and synthesize the analyses provided by the
                Financial Analyst and the Research Analyst.
                Combine these insights to form a comprehensive
                investment recommendation. 
                
                You MUST Consider all aspects, including financial
                health, market sentiment, and qualitative data from
                EDGAR filings.

                Make sure to include a section that shows insider 
                trading activity, and upcoming events like earnings.

                Your final answer MUST be a recommendation for your
                customer. It should be a full super detailed report, providing a 
                clear investment stance and strategy with supporting evidence.
                Make it pretty and well formatted for your customer.
                """
            ),
            expected_output="A complete investment recommendation",
            tools=[tool],
            agent=investment_advisor,
            dependencies=[research_task, financial_analysis_task, filings_analysis_task]
        )

        recommend_task = Task(
            description=dedent(
                f"""
                Review and synthesize all analyses provided by the 
                Financial Analyst, Research Analyst, and Investment Advisor. 
                Focus on creating a final investment strategy that aligns 
                with the customer's goals and preferences. 

                Use all aspects, including financial health, market sentiment, 
                qualitative data from EDGAR filings, and expert recommendations.

                Your final answer MUST be a full comprehensive recommendation 
                report. Include key points from each analysis, your final 
                recommendation on the stock (buy, hold, or sell), and a 
                suggested investment strategy tailored to the customer's needs.
                
                Make the report visually appealing and provide clear evidence 
                to back your recommendations.
                """
            ),
            expected_output=dedent(
                f"""Your final answer MUST be a comprehensive recommendation report, 
                including key points from each analysis, a final recommendation 
                on the stock (buy, hold, or sell), and a suggested investment 
                strategy tailored to the customer's needs."""
            ),
            tools=[tool],
            agent=investment_advisor,
            dependencies=[research_task, financial_analysis_task, filings_analysis_task, investment_advice_task]  # Runs after all other tasks
        )

        crew = Crew(
            agents=[research_analyst, financial_analyst, investment_advisor],
            tasks=[research_task, financial_analysis_task, filings_analysis_task, investment_advice_task, recommend_task],
            verbose=True
        )

        result = crew.kickoff(inputs={'stock_symbol': stock_symbol})

        return str(result)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
