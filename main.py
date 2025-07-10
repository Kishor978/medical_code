from openai import OpenAI
from tool_handler import ToolHandler
from tools import get_icd_10_parser_tool
from system_prompt import system_prompt
import asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4o"

from openai import OpenAI
import os

class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def get_response(self, messages, system_prompt, tools=None):
        loop = asyncio.get_event_loop()
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=full_messages,
                max_tokens=500,
                stream=False
            )
        )
        return response

class AgenticLoop:
    def __init__(self, messages: list[dict], tools: list[dict], system_prompt: str):
        self.messages = messages
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_handler = ToolHandler()
        self.llm = LLM()

    async def run_tool_call(self, tool_use):
        try:
            print(f"\n=== Using {tool_use.function.name} tool ===")
            tool_result = self.tool_handler.process_tool_call(tool_use.function)
            return {
                "role": "tool",
                "tool_call_id": tool_use.id,
                "content": str(tool_result)
            }
        except Exception as e:
            error_msg = f"Error executing tool {tool_use.function.name}: {str(e)}"
            print(error_msg)
            return {
                "role": "tool",
                "tool_call_id": tool_use.id,
                "content": error_msg
            }

    async def run_requested_tools(self, response):
        tool_calls = response.choices[0].message.tool_calls or []
        if tool_calls:
            tool_results = await asyncio.gather(
                *[self.run_tool_call(tool_use) for tool_use in tool_calls],
                return_exceptions=False
            )
            return tool_results
        return "No tool calls requested"

    async def generate_response(self):
        print('fetching response')
        response = await self.llm.get_response(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=self.tools
        )

        assistant_message = response.choices[0].message
        self.messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })

        tool_results = await self.run_requested_tools(response)

        if tool_results != "No tool calls requested":
            self.messages.extend(tool_results)
            return await self.generate_response()

        return response

async def main():
    default_note = """PATIENT NOTE
Date: 2024-03-20
RE: Initial Visit

CHIEF COMPLAINT:
Patient presents with symptoms of Type 2 diabetes mellitus with early signs of diabetic neuropathy in both feet. 
Also reports ongoing hypertension.

HISTORY OF PRESENT ILLNESS:
52-year-old male reports increased thirst, frequent urination, and numbness/tingling in feet for past 3 months. 
Blood sugar readings at home consistently above 200 mg/dL. Has family history of diabetes (mother and sister). 
Patient also notes ongoing high blood pressure despite current medication.

VITAL SIGNS:
- BP: 142/90 mmHg
- Pulse: 78
- Weight: 198 lbs
- Height: 5'10"
- BMI: 28.4

LAB RESULTS:
- Fasting Blood Glucose: 186 mg/dL
- HbA1c: 7.8%

MEDICATIONS:
- Lisinopril 10mg daily for hypertension
- No current diabetes medications

ASSESSMENT:
1. Type 2 diabetes mellitus, uncontrolled
2. Diabetic neuropathy
3. Essential hypertension

PLAN:
- Start Metformin 500mg twice daily
- Continue Lisinopril
- Diabetes education referral
- Follow-up in 2 weeks
- Recommend diet and exercise program"""

    print("\nPlease enter the patient notes (press Enter twice to use default note):")

    user_input = []
    while True:
        line = input()
        if line == "" and (len(user_input) == 0 or user_input[-1] == ""):
            break
        user_input.append(line)

    patient_note = "\n".join(user_input) if user_input and user_input != [''] else default_note

    initial_messages = [{
        "role": "user",
        "content": f"Please analyze these patient notes and identify the appropriate ICD-10 codes:\n\n{patient_note}"
    }]

    tools = [get_icd_10_parser_tool()]

    agent = AgenticLoop(
        messages=initial_messages,
        tools=tools,
        system_prompt=system_prompt
    )

    response = await agent.generate_response()
    print("Final response:", response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
