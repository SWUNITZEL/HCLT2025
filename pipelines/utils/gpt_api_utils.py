import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai


load_dotenv()

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Claude 설정
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# Gemini 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# 클라이언트 초기화
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

def load_prompt(file_path, **kwargs):
    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)

def append_reasoning_instruction(user_prompt, reasoning):
    """reasoning이 True면 단계별 사고 지시문을 붙여줌"""
    if reasoning:
        reasoning_instruction = "\n\nPlease think step by step and provide detailed reasoning for your answers."
        return user_prompt + reasoning_instruction
    return user_prompt

def call_gpt(system_prompt, user_prompt, temperature=TEMPERATURE, reasoning=False):
    """OpenAI GPT API 호출"""
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT API 호출 오류: {e}")
        return ""

def call_gpt4_with_model(system_prompt, user_prompt, model_name, temperature=TEMPERATURE, reasoning=False):
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT API 호출 오류 (모델: {model_name}): {e}")
        return ""

def call_gpt5_with_model(system_prompt, user_prompt, model_name, temperature=TEMPERATURE, reasoning=False):
    try:
        print(f"    GPT5 API 호출 시작 - 모델: {model_name}, reasoning: {reasoning}")
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        result = response.choices[0].message.content
        
        print(f"    GPT5 API 호출 성공 - 응답 길이: {len(result)}")
        return result
    except Exception as e:
        print(f"GPT5 API 호출 오류 (모델: {model_name}): {e}")
        return ""

def call_claude(system_prompt, user_prompt, temperature=TEMPERATURE, reasoning=False):
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude API 호출 오류: {e}")
        return ""

def call_claude_with_model(system_prompt, user_prompt, model_name, temperature=TEMPERATURE, reasoning=False):
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        response = claude_client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude API 호출 오류 (모델: {model_name}): {e}")
        return ""

def call_gemini(system_prompt, user_prompt, temperature=TEMPERATURE, reasoning=False):
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini API 호출 오류: {e}")
        return ""

def call_gemini_with_model(system_prompt, user_prompt, model_name, temperature=TEMPERATURE, reasoning=False):
    try:
        user_prompt = append_reasoning_instruction(user_prompt, reasoning)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=1.0,
                max_output_tokens=8192
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini API 호출 오류 (모델: {model_name}): {e}")
        return ""
