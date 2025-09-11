# 파일명: multi_model_evaluator.py

import os
import json
import numpy as np
import time
from typing import List, Dict, Any

# --- 사전 준비 ---
from utils.gpt_api_utils import (
    call_gpt4_with_model, 
    call_gpt5_with_model, 
    call_claude_with_model, 
    call_gemini_with_model
)
from rouge_score import rouge_scorer

class MultiModelEvaluator:
    """
    미리 통합된 300개의 QA 세트 파일을 사용하여,
    student_agent.txt 프롬프트 템플릿으로 여러 LLM 모델을 순차적으로 평가하는 파이프라인.
    """
    def __init__(self):
        self.eval_dir = "data/qa/unified_eval_results_300"
        self.unified_data_path = os.path.join(self.eval_dir, "unified_300_qa_sets.json")
        os.makedirs(self.eval_dir, exist_ok=True)

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

        # ✅ 1. student_agent.txt를 로드하고 system/user 템플릿으로 분리하여 저장합니다.
        self.system_template, self.user_template = self._load_and_split_prompt_template()

        # --- ✅ 평가할 모델과 설정 정의 ---
        self.models_to_evaluate = {
            "GPT4o_on": { "func": call_gpt4_with_model, "model_name": "gpt-4o", "reasoning": True },
            "GPT4o_off": { "func": call_gpt4_with_model, "model_name": "gpt-4o-mini", "reasoning": False },
        }

    # ✅ 2. student_agent.txt를 로드하고 분리하는 헬퍼 함수
    def _load_and_split_prompt_template(self) -> (str, str):
        prompt_path = os.path.join("config", "prompts", "student_agent.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"프롬프트 파일이 없습니다: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            full_template = f.read()

        if '---' not in full_template:
            raise ValueError("프롬프트 파일에 system과 user 프롬프트를 구분하는 '---' 마커가 없습니다.")
        
        system_part, user_part = full_template.split('---', 1)
        return system_part.strip(), user_part.strip()

    def load_unified_data(self) -> List[Dict[str, Any]]:
        """통합된 QA 세트 JSON 파일을 로드합니다."""
        print(f"통합 데이터 파일 로드 중: '{self.unified_data_path}'")
        if not os.path.exists(self.unified_data_path):
            raise FileNotFoundError(f"오류: 통합 데이터 파일이 없습니다. 먼저 prepare_dataset.py를 실행하세요.")
        
        with open(self.unified_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ✅ 3. dispatch_api_call 함수가 분리된 템플릿을 사용하도록 수정
    def dispatch_api_call(self, model_info: Dict[str, Any], qa_item: Dict[str, Any]) -> str:
        """설정에 맞는 모델의 API를 호출하고 응답을 출력합니다."""
        api_func = model_info["func"]
        model_name = model_info["model_name"]
        reasoning = model_info["reasoning"]
        
        # qa_item에서 필요한 모든 정보 추출
        question = qa_item["question"]
        department = qa_item.get("department", "해당 학과")
        document = qa_item.get("document", "제공된 문서 없음")
        
        # 시스템 프롬프트 템플릿 채우기
        system_prompt = self.system_template.format(department=department)

        # 유저 프롬프트 템플릿 채우기
        # 단일 질문을 {questions} 플레이스홀더 형식에 맞게 변환
        questions_text = f"1. {question}"
        user_prompt = self.user_template.format(questions=questions_text, document=document)
        
        print(f"    모델 호출: {model_name} (reasoning={'on' if reasoning else 'off'})")
        print(f"    질문: {question[:100]}...")
        
        try:
            answer = api_func(system_prompt, user_prompt, model_name, reasoning=reasoning)
            
            refusal_keywords = ["죄송", "할 수 없습니다", "요청을 수행할 수 없습니다", "I cannot fulfill", "I'm unable to"]
            if any(keyword in answer for keyword in refusal_keywords):
                print(f"    ⚠️ 경고: 모델이 답변을 거부했습니다. 응답: {answer[:150]}...")
            else:
                print(f"    --> 모델 응답: {answer[:200]}...")
            
            return answer
        except Exception as e:
            print(f"API 호출 중 오류 발생 ({model_name}, reasoning={'on' if reasoning else 'off'}): {e}")
            return "[API ERROR]"

    def calculate_max_rouge_score(self, generated_answer: str, ground_truths: List[str]) -> Dict[str, float]:
        """하나의 생성된 답변과 여러 정답 후보 간의 ROUGE 점수 중 가장 높은 F1 점수를 반환합니다."""
        if not generated_answer or not ground_truths:
            return {"rouge1_f1": 0.0, "rougeL_f1": 0.0}

        scores = [self.rouge_scorer.score(gt, generated_answer) for gt in ground_truths]
        max_rouge1_f1 = max([s['rouge1'].fmeasure for s in scores]) if scores else 0.0
        max_rougeL_f1 = max([s['rougeL'].fmeasure for s in scores]) if scores else 0.0
        
        return {"rouge1_f1": max_rouge1_f1, "rougeL_f1": max_rougeL_f1}

    # ✅ 4. run_full_evaluation 함수는 qa_item 전체를 넘겨주므로 수정할 필요 없음
    def run_full_evaluation(self):
        """정의된 모든 모델과 설정에 대해 전체 평가를 실행합니다."""
        all_qa_sets = self.load_unified_data()
        total_sets = len(all_qa_sets)
        overall_summary = {}

        for run_key, model_info in self.models_to_evaluate.items():
            print(f"\n{'='*20}\n🚀 '{run_key}' 평가를 시작합니다... ({total_sets}개 질문)\n{'='*20}")

            run_results = []
            all_rouge1_f1 = []
            all_rougeL_f1 = []

            for i, qa_item in enumerate(all_qa_sets):
                print(f"  -> {run_key}: 질문 {i+1}/{total_sets} 처리 중...")
                
                generated_answer = self.dispatch_api_call(model_info, qa_item)
                scores = self.calculate_max_rouge_score(generated_answer, qa_item["ground_truths"])
                
                run_results.append({
                    "unified_id": qa_item["unified_id"],
                    "question": qa_item["question"],
                    "generated_answer": generated_answer,
                    "scores": scores
                })
                all_rouge1_f1.append(scores["rouge1_f1"])
                all_rougeL_f1.append(scores["rougeL_f1"])
                
                time.sleep(1)

            avg_rouge1_f1 = np.mean(all_rouge1_f1)
            avg_rougeL_f1 = np.mean(all_rougeL_f1)

            overall_summary[run_key] = {
                "ROUGE-1_F1_avg": avg_rouge1_f1,
                "ROUGE-L_F1_avg": avg_rougeL_f1,
            }

            detailed_filename = os.path.join(self.eval_dir, f"detailed_results_{run_key}.json")
            with open(detailed_filename, "w", encoding="utf-8") as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
            
            print(f"'{run_key}' 평가 완료! 평균 ROUGE-1 F1: {avg_rouge1_f1:.4f}, ROUGE-L F1: {avg_rougeL_f1:.4f}")

        summary_filename = os.path.join(self.eval_dir, "evaluation_summary_all_models.json")
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)

        print("\n\n--- 🏆 최종 평가 요약 🏆 ---")
        for run_name, scores in overall_summary.items():
            print(f"  - {run_name}:")
            print(f"    - ROUGE-1 F1 평균: {scores['ROUGE-1_F1_avg']:.4f}")
            print(f"    - ROUGE-L F1 평균: {scores['ROUGE-L_F1_avg']:.4f}")
        print(f"\n모든 평가가 완료되었습니다. 상세 결과는 '{self.eval_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    evaluator = MultiModelEvaluator()
    evaluator.run_full_evaluation()