import re
from typing import List
from utils.gpt_api_utils import call_gpt, load_prompt
class GroundTruthAgent:
    def __init__(self, prompt_path="config/prompts/ground_truth_agent.txt"):
        self.prompt_path = prompt_path

    def generate_ground_truth(self, department: str, document: str, questions: List[str]):
        """
        생성된 질문에 ground truth 생성
        """
        # 프롬프트 불러오기 + 변수 치환
        prompt = load_prompt(
            self.prompt_path,
            department=department,
            document=document,
            questions=questions
        )

        system_prompt, user_prompt= prompt.split("---", 1)
        
        # ChatGPT API 호출
        result = call_gpt(system_prompt, user_prompt)
        
        # 정규식으로 groundtruth 전처리
        pattern = re.compile(r'\(?(\d+)\)?\s*:\s*\[(.*?)\]', re.DOTALL)
        matches = pattern.findall(result)

        ground_truth = {}
        for key, val in matches:
            answers = [m.strip().strip('()"') for m in val.split('.,') if m.strip()]
            if len(answers) < 2:
                answers = re.findall(r'["“](.*?)["”]', val, re.DOTALL)
                answers = [a.strip() for a in answers if a.strip()]    
            ground_truth[key] = answers
            
        return ground_truth