from utils.gpt_api_utils import call_gpt
from typing import List

class StudentAgent:
    def __init__(self):
        pass

    def generate_student_answer(self, department: str, document: str, questions: List[str]):
        """
        학생처럼 답변을 생성하는 에이전트
        """
        # 질문들을 문자열로 변환
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        # 직접 프롬프트 정의
        system_prompt = f"""You are a high school student in South Korea with logical thinking skills.
You are applying for admission to the {department} department, and this is an admissions interview situation.
Create ideal interview answers for each following question.
When generating answers, follow the answer generation rules.
Answer only in Korean, following the answer format strictly.
Do not include any explanation, summary, or additional text beyond the ranked list."""

        user_prompt = f"""Following questions:
{questions_text}

Answer generation rules:
- Your answers must remain within the scope of the Korean high school curriculum.
- Begin with the main point (top-down structure).
- For experience-related questions, answers should be concise and authentic; avoid exaggeration or fabrication, and refer to the provided document when necessary.
- Each answer should last approximately 40 seconds to 50 seconds, which corresponds to 3 to 4 sentences in Korean.
- Refer to the structure of the provided sample answers when formulating responses.
- Each question should include one answer.

Document:
{document}

Answer list format (no extra text):
(question index): (answer)"""

        # ChatGPT API 호출
        student_answers = call_gpt(system_prompt, user_prompt)
        
        return student_answers


