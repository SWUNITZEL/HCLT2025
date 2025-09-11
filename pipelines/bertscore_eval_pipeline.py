import os
import json
import numpy as np
from typing import List, Dict, Any
from agents.student_agent import StudentAgent
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

class BERTScoreEvalPipeline:
    def __init__(self):
        self.qa_dir_path = "data/qa"
        self.ground_truth_1_dir_path = os.path.join(self.qa_dir_path, "ground_truth_1")
        self.student_agent = StudentAgent()
        
        # 결과 저장 디렉토리 생성
        self.eval_dir_path = os.path.join(self.qa_dir_path, "eval")
        os.makedirs(self.eval_dir_path, exist_ok=True)
        
    def load_qa_data(self, qa_id: str) -> Dict[str, Any]:
        """ground_truth_1에서 QA 데이터를 로드합니다."""
        qa_file_path = os.path.join(self.ground_truth_1_dir_path, f"qa_{qa_id}.json")
        
        if not os.path.exists(qa_file_path):
            raise FileNotFoundError(f"QA 파일을 찾을 수 없습니다: {qa_file_path}")
        
        with open(qa_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def extract_questions(self, qa_data: Dict[str, Any]) -> List[str]:
        """QA 데이터에서 질문들을 추출합니다."""
        questions = []
        
        # qa_data의 구조에 따라 질문 추출
        if "qa" in qa_data:
            for qa_item in qa_data["qa"]:
                if "question" in qa_item:
                    questions.append(qa_item["question"])
        else:
            # 다른 구조일 경우 처리
            for key, value in qa_data.items():
                if isinstance(value, dict) and "qa" in value:
                    for qa_item in value["qa"]:
                        if "question" in qa_item:
                            questions.append(qa_item["question"])
        
        return questions
    
    def extract_ground_truth_answers(self, qa_data: Dict[str, Any]) -> List[List[str]]:
        """QA 데이터에서 ground truth 답변들을 추출합니다."""
        ground_truth_answers = []
        
        # qa_data의 구조에 따라 ground truth 추출
        if "qa" in qa_data:
            for qa_item in qa_data["qa"]:
                if "ground_truth" in qa_item:
                    ground_truth_answers.append(qa_item["ground_truth"])
        else:
            # 다른 구조일 경우 처리
            for key, value in qa_data.items():
                if isinstance(value, dict) and "qa" in value:
                    for qa_item in value["qa"]:
                        if "ground_truth" in qa_item:
                            ground_truth_answers.append(qa_item["ground_truth"])
        
        return ground_truth_answers
    
    def generate_student_answers(self, department: str, document: str, questions: List[str]) -> List[str]:
        """StudentAgent를 사용하여 학생 답변을 생성합니다."""
        try:
            student_answers_text = self.student_agent.generate_student_answer(
                department=department,
                document=document,
                questions=questions
            )
            
            # 답변 텍스트를 파싱하여 리스트로 변환
            student_answers = self.parse_student_answers(student_answers_text, len(questions))
            return student_answers
            
        except Exception as e:
            print(f"학생 답변 생성 중 오류 발생: {e}")
            return [""] * len(questions)
    
    def parse_student_answers(self, answers_text: str, num_questions: int) -> List[str]:
        """학생 답변 텍스트를 파싱하여 리스트로 변환합니다."""
        answers = []
        lines = answers_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('(')):
                # 숫자나 괄호로 시작하는 라인에서 답변 추출
                if ':' in line:
                    answer = line.split(':', 1)[1].strip()
                    if answer.startswith('(') and answer.endswith(')'):
                        answer = answer[1:-1].strip()
                    answers.append(answer)
                elif ')' in line:
                    # (숫자): 답변 형태 처리
                    if ':' in line:
                        answer = line.split(':', 1)[1].strip()
                        answers.append(answer)
        
        # 답변 개수가 맞지 않으면 빈 문자열로 채움
        while len(answers) < num_questions:
            answers.append("")
        
        return answers[:num_questions]
    
    def calculate_bertscore(self, student_answer: str, ground_truth_answers: List[str]) -> Dict[str, float]:
        """하나의 학생 답변과 여러 ground truth 답변 간의 BERTScore를 계산합니다."""
        if not student_answer.strip():
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # 각 ground truth와의 BERTScore 계산
            scores = []
            for gt_answer in ground_truth_answers:
                if gt_answer.strip():
                    P, R, F1 = bert_score([student_answer], [gt_answer], 
                                        model_type='distilbert-base-multilingual-cased',
                                        lang='ko', verbose=False)
                    scores.append({
                        "precision": P.item(),
                        "recall": R.item(), 
                        "f1": F1.item()
                    })
            
            if not scores:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # 평균 계산
            avg_scores = {
                "precision": np.mean([s["precision"] for s in scores]),
                "recall": np.mean([s["recall"] for s in scores]),
                "f1": np.mean([s["f1"] for s in scores])
            }
            
            return avg_scores
            
        except Exception as e:
            print(f"BERTScore 계산 중 오류 발생: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def evaluate_qa(self, qa_id: str) -> Dict[str, Any]:
        """하나의 QA 데이터에 대해 평가를 수행합니다."""
        print(f"QA {qa_id} 평가 시작...")
        
        # QA 데이터 로드
        qa_data = self.load_qa_data(qa_id)
        
        # 질문과 ground truth 추출
        questions = self.extract_questions(qa_data)
        ground_truth_answers = self.extract_ground_truth_answers(qa_data)
        
        if not questions or not ground_truth_answers:
            print(f"QA {qa_id}: 질문 또는 ground truth를 찾을 수 없습니다.")
            return {}
        
        # department와 document 정보 추출
        department = qa_data.get("department", "사학과")
        document = qa_data.get("document", "")
        
        # 학생 답변 생성
        student_answers = self.generate_student_answers(department, document, questions)
        
        # 각 질문에 대한 BERTScore 계산
        question_scores = []
        for i, (question, student_answer, gt_answers) in enumerate(zip(questions, student_answers, ground_truth_answers)):
            scores = self.calculate_bertscore(student_answer, gt_answers)
            
            question_scores.append({
                "question_index": i + 1,
                "question": question,
                "student_answer": student_answer,
                "ground_truth_answers": gt_answers,
                "bertscore": scores
            })
            
            print(f"질문 {i+1} BERTScore - Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F1: {scores['f1']:.4f}")
        
        # 전체 평균 계산
        avg_precision = np.mean([qs["bertscore"]["precision"] for qs in question_scores])
        avg_recall = np.mean([qs["bertscore"]["recall"] for qs in question_scores])
        avg_f1 = np.mean([qs["bertscore"]["f1"] for qs in question_scores])
        
        evaluation_result = {
            "qa_id": qa_id,
            "department": department,
            "total_questions": len(questions),
            "question_scores": question_scores,
            "average_scores": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
        }
        
        print(f"QA {qa_id} 전체 평균 - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        
        return evaluation_result
    
    def run_evaluation(self, qa_ids: List[str] = None):
        """전체 평가를 실행합니다."""
        if qa_ids is None:
            # ground_truth_1 디렉토리에서 모든 QA 파일 찾기
            qa_files = [f for f in os.listdir(self.ground_truth_1_dir_path) if f.startswith("qa_") and f.endswith(".json")]
            qa_ids = [f.replace("qa_", "").replace(".json", "") for f in qa_files]
        
        all_results = []
        
        for qa_id in qa_ids:
            try:
                result = self.evaluate_qa(qa_id)
                if result:
                    all_results.append(result)
                    
                    # 개별 결과 저장
                    result_file_path = os.path.join(self.eval_dir_path, f"eval_qa_{qa_id}.json")
                    with open(result_file_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                        
            except Exception as e:
                print(f"QA {qa_id} 평가 중 오류 발생: {e}")
        
        # 전체 결과 저장
        if all_results:
            overall_result = {
                "evaluation_summary": {
                    "total_qa_evaluated": len(all_results),
                    "qa_ids": qa_ids
                },
                "detailed_results": all_results,
                "overall_averages": {
                    "precision": np.mean([r["average_scores"]["precision"] for r in all_results]),
                    "recall": np.mean([r["average_scores"]["recall"] for r in all_results]),
                    "f1": np.mean([r["average_scores"]["f1"] for r in all_results])
                }
            }
            
            overall_file_path = os.path.join(self.eval_dir_path, "overall_evaluation.json")
            with open(overall_file_path, "w", encoding="utf-8") as f:
                json.dump(overall_result, f, ensure_ascii=False, indent=4)
            
            print(f"\n전체 평가 완료!")
            print(f"평가된 QA 수: {len(all_results)}")
            print(f"전체 평균 F1: {overall_result['overall_averages']['f1']:.4f}")
        
        return all_results


