import re
from typing import List
from utils.gpt_api_utils import call_gpt, load_prompt
class PriorityAgent:
    def __init__(self, prompt_path="config/prompts/priority_agent.txt"):
        self.prompt_path = prompt_path

    def generate_priority(self, department: str, questions: List[str]):
        """
        생성된 질문에 우선순위 규칙 적용
        """
        # 프롬프트 불러오기 + 변수 치환
        prompt = load_prompt(
            self.prompt_path,
            department=department,
            questions=questions,
        )

        system_prompt, user_prompt= prompt.split("---", 1)
        
        # ChatGPT API 호출
        result = call_gpt(system_prompt, user_prompt)
        questions = [{
                        "level": re.search(r'\b\d+\b', re.sub(r'^\d+\.\s*', '', question).strip()).group(),
                        "ranking": re.sub(r'^(\d+)\..*', r'\1', question).strip(),
                        "question": re.sub(r'^\(?\d+\)?\.?\s*', '', question).split(":",maxsplit=1)[1].strip().split(maxsplit=1)[1].strip()
                        } 
                     for question in result.split("\n") if question.strip() != ""]
        return questions