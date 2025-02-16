from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
from tools import tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please set GEMINI_API_KEY environment variable.")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/proxy/*": {"origins": "*"}})  # Enable CORS for proxy routes

# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.5,
    max_tokens=1024,
    top_p=0.9,
    api_key=api_key
)

# Define Agents
financial_analyst = Agent(
    role="The Best Financial Analyst",
    goal="Provide deep financial analysis and insights.",
    backstory="A seasoned financial analyst specializing in stock market trends.",
    llm=llm,
    verbose=True
)

research_analyst = Agent(
    role="Staff Research Analyst",
    goal="Gather and interpret market and financial data effectively.",
    backstory="A highly skilled research analyst specializing in market sentiment.",
    llm=llm,
    verbose=True
)

investment_advisor = Agent(
    role="Private Investment Advisor",
    goal="Provide clients with expert investment advice based on data-driven insights.",
    backstory="An experienced investment advisor with a strong track record.",
    llm=llm,
    verbose=True
)

# Proxy route to handle frontend requests
@app.route("/proxy/analyze", methods=["POST", "OPTIONS"])
def proxy_analyze():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    try:
        data = request.get_json()
        stock_symbol = data.get('company')

        if not stock_symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        # Define tasks
        research_task = Task(
            description=f"""
                Gather and summarize recent news, press releases, and 
                market analyses related to {stock_symbol}. Highlight 
                significant events, trends, and analyst opinions.
            """,
            expected_output="A comprehensive summary report",
            tools=[tool],
            agent=research_analyst
        )

        financial_analysis_task = Task(
            description=f"""
                Conduct a financial health analysis for {stock_symbol}, 
                including metrics like P/E ratio, EPS growth, and revenue trends.
            """,
            expected_output="A detailed financial analysis",
            tools=[tool],
            agent=financial_analyst
        )

        investment_advice_task = Task(
            description=f"""
                Synthesize financial and research data to provide a 
                strategic investment recommendation for {stock_symbol}.
            """,
            expected_output="A comprehensive investment strategy",
            tools=[tool],
            agent=investment_advisor,
            dependencies=[research_task, financial_analysis_task]
        )

        # Create Crew
        crew = Crew(
            agents=[research_analyst, financial_analyst, investment_advisor],
            tasks=[research_task, financial_analysis_task, investment_advice_task],
            verbose=True
        )

        result = crew.kickoff(inputs={'stock_symbol': stock_symbol})
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
