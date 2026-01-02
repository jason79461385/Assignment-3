import sys
import os
import datetime
import warnings
import re
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", message=".*Convert_system_message_to_human.*")
warnings.filterwarnings("ignore", message=".*API key must be provided when using hosted LangSmith API.*")
import time
from termcolor import colored
from graph_agent import run_graph_agent, run_legacy_agent
from config import get_llm
from langchain_core.prompts import ChatPromptTemplate

TEST_MODE = "LEGACY" # Options: "GRAPH" or "LEGACY"

class DualLogger:
    def __init__(self, filename="evaluation_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message) 
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)      
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def grade_answer_with_llm(question, agent_answer, expected_facts, forbidden_facts):
    llm = get_llm(temperature=0) 
    
    prompt = ChatPromptTemplate.from_template("""
    You are a strict grading assistant. Check if the AGENT_ANSWER meets the following criteria based on the QUESTION.
    
    QUESTION: {question}
    AGENT_ANSWER: {agent_answer}
    
    CRITERIA 1 (Must Include): The answer MUST semantically contain these facts: {expected_facts}.
    CRITERIA 2 (Forbidden): The answer MUST NOT contain information about: {forbidden_facts}.
    
    Logic:
    - If the agent says "391 billion" and expected is "391,000 million", that is PASS.
    - If the agent answers in a different language but the numbers/facts are correct, that is PASS.
    - If the agent hallucinates or mentions forbidden topics (e.g., mentioning Tesla when asked about Apple), that is FAIL.
    
    OUTPUT ONLY ONE WORD: "PASS" or "FAIL".
    """)
    
    chain = prompt | llm
    result = chain.invoke({
        "question": question, 
        "agent_answer": agent_answer,
        "expected_facts": str(expected_facts),
        "forbidden_facts": str(forbidden_facts)
    })
    
    return result.content.strip().upper()

TEST_CASES = [
    {
        "name": "Test A: Apple Revenue",
        "question": "Apple 2024 年的總營收 (Total net sales) 是多少？",
        "must_contain": ["391", "billion", "3,910", "億"],
        "forbidden": ["Tesla"]
    },
    {
        "name": "Test B: Tesla R&D",
        "question": "Tesla 2024 年的研發費用 (R&D expenses) 是多少？",
        "must_contain": ["4.77", "47.7", "billion"],
        "forbidden": ["Apple"]
    },
    
    {
        "name": "Test C: R&D Comparison",
        "question": "比較 Apple 和 Tesla 2024 年的研發支出，誰比較高？",
        "must_contain": ["Apple", "Tesla", "31", "4.7", "higher"],
        "forbidden": []
    },

    {
        "name": "Test D: Apple Services Cost",
        "question": "Apple 2024 年的「服務成本 (Cost of sales - Services)」是多少？",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden": []
    },
    {
        "name": "Test E: Tesla Energy Revenue",
        "question": "Tesla 2024 年的「能源發電與儲存 (Energy generation and storage)」營收是多少？",
        "must_contain": ["23.7", "billion", "23,767"],
        "forbidden": []
    },

    {
        "name": "Test F: Semantic Trap (Profitability)",
        "question": "這兩家公司在 2024 年誰比較會賺錢？(比較 Net Income)",
        "must_contain": ["Apple", "Tesla", "93", "7", "billion"],
        "forbidden": []
    },

    {
        "name": "Test G: Unknown Info",
        "question": "Apple 計畫在 2025 年發布的 iPhone 17 預計售價是多少？",
        "must_contain": ["unknown", "provide", "mention", "does not", "無法", "未提及"],
        "forbidden": ["1000", "999", "1200"] 
    },
    {
        "name": "Test A1 [Eng]: Apple Revenue",
        "question": "What was Apple's Total Net Sales for the fiscal year 2024?",
        "must_contain": ["391", "billion", "391,035"], 
        "forbidden": ["Tesla"]
    },
    {
        "name": "Test A2 [Eng]: Tesla Automotive Revenue",
        "question": "What is the specific revenue figure for 'Automotive sales' for Tesla in 2024?",
        "must_contain": ["78", "billion", "78,512"], 
        "forbidden": ["Apple"]
    },

    {
        "name": "Test B1 [Mixed]: Apple R&D",
        "question": "Apple 2024 年的研發費用 (Research and development expenses) 是多少？",
        "must_contain": ["31", "billion", "31,370"], 
        "forbidden": ["Tesla"]
    },
    {
        "name": "Test B2 [Mixed]: Tesla CapEx",
        "question": "Tesla 在 2024 年的資本支出 (Capital Expenditures) 是多少？",
        "must_contain": ["11", "billion", "11,153"], 
        "forbidden": ["Apple"]
    },
    {
        "name": "Test C1 [Eng]: R&D Comparison",
        "question": "Compare the Research and Development (R&D) expenses of Apple and Tesla in 2024. Who spent more?",
        "must_contain": ["Apple", "Tesla", "31", "4.7", "Apple spent more", "higher"],
        "forbidden": [] 
    },
    {
        "name": "Test C2 [Eng]: Gross Margin Analysis",
        "question": "Which company had a higher Total Gross Margin percentage in 2024, Apple or Tesla? Please provide the approximate percentages.",
        "must_contain": ["Apple", "Tesla", "46", "18", "Apple"],
        "forbidden": []
    },

    {
        "name": "Test D1 [Eng]: Apple Services Cost",
        "question": "According to the Consolidated Statements of Operations, what was Apple's 'Cost of sales' specifically for 'Services' in 2024?",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden": []
    },
    
    {
        "name": "Test E1 [Mixed]: 2025 Projection (Trap)",
        "question": "財報中有提到 Apple 2025 年預計的 iPhone 銷量目標嗎？",
        "must_contain": ["no", "not mentioned", "does not provide", "沒有提到", "未知"], 
        "forbidden": ["100 million", "200 million", "increase"] # 禁止瞎掰數字
    },
    
    {
        "name": "Test F1 [Eng]: CEO Identity",
        "question": "Who signed the 10-K report as the Chief Executive Officer for Tesla?",
        "must_contain": ["Elon Musk"],
        "forbidden": ["Tim Cook"]
    }
]

def run_evaluation():
    score = 0
    total = len(TEST_CASES)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==================================================")
    print(f"📄 ASSIGNMENT 3 EVALUATION REPORT")
    print(f"⏰ Time: {timestamp}")
    print(f"🤖 Agent Mode: {TEST_MODE}")
    print(f"==================================================\n")

    print(colored("🚀 STARTING AI-POWERED EVALUATION...", "cyan", attrs=["bold"]))
    print(f"==================================================\n")


    for test in TEST_CASES:
        print(f"Running: {test['name']}...")
        start_time = time.time()
        
        try:
            if TEST_MODE == "GRAPH":
                answer = run_graph_agent(test["question"])
            else:
                answer = run_legacy_agent(test["question"])
            
            result = grade_answer_with_llm(
                test["question"], 
                answer, 
                test["must_contain"], 
                test["forbidden"]
            )
            
            elapsed = time.time() - start_time
            print(f"A: {answer.strip()}")
            if "PASS" in result:
                score += 1
                print(colored(f"✅ PASS ({elapsed:.2f}s)", "green"))
            else:
                print(colored(f"❌ FAIL ({elapsed:.2f}s)", "red"))
                print(f"   Agent Answer: {answer}")
                print(f"   Judge Verdict: {result}")

        except Exception as e:
            print(colored(f"❌ CRASH: {e}", "red"))
        print("-" * 50)

    print(colored(f"\n📊 FINAL SCORE: {score}/{total}", "magenta", attrs=["bold"]))

if __name__ == "__main__":
    log_filename = f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    sys.stdout = DualLogger(log_filename)
    run_evaluation()
    sys.stdout = sys.stdout.terminal
    print(f"\n[System] Log saved to {log_filename}")