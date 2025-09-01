import os
import json
import traceback
from agents.document_agent import DocumentAgent
from agents.comment_agent import CommentAgent
from agents.question_gen_agent import QuestionGenAgent
from agents.priority_agent import PriorityAgent
from agents.ground_truth_agent import GroundTruthAgent
class GroundTruthGenPipeline:
    def __init__(self, base_path: str):
        self.document_agent = DocumentAgent()
        self.comment_agent = CommentAgent()
        self.question_gen_agent = QuestionGenAgent()
        self.priority_agent = PriorityAgent()
        self.ground_truth_agent = GroundTruthAgent()
        
        self.processed_dir_path = os.path.join(base_path, "data", "processed")
        if not os.path.exists(self.processed_dir_path):
            os.makedirs(self.processed_dir_path)
        self.qa_dir_path = os.path.join(base_path, "data", "qa")
        if not os.path.exists(self.qa_dir_path):
            os.makedirs(self.qa_dir_path)
        self.eval_dir_path = os.path.join(base_path, "data", "eval")
        if not os.path.exists(self.eval_dir_path):
            os.makedirs(self.eval_dir_path)
        
    def run(self, id: str, department: str, document: str):
        """
        department, document를 입력 받아 pipeline 실행
        """
        processed_json_path = os.path.join(self.processed_dir_path, f"processed_{id}.json")
        qa_ground_truth_json_path = os.path.join(self.qa_dir_path, "ground_truth", f"qa_{id}.json")
        eval_json_path = os.path.join(self.eval_dir_path, f"eval_{id}.json")
        try:
            # 개별 processed.json 생성/불러오기
            if not os.path.exists(processed_json_path):
                processed_data={}
                processed_data[id] = {
                    "department":department,
                    "document":document
                    }
            else:
                with open(processed_json_path, "r", encoding="utf-8") as f:
                    processed_data = json.load(f)
                    
            summary = self.document_agent.generate_document(
                department=department,
                document=document
            )
            processed_data[id]["summary"]=summary
            
            # comment가 없을 경우 gpt로 값 생성
            if  not processed_data[id].get("comment"):
                # comment 생성
                comment = self.comment_agent.generate_comment(
                    department=department,
                    document=summary
                )
                processed_data[id]["comment"]=comment
            else:
                comment = processed_data[id]["comment"]
            
            with open(processed_json_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            
            # 질문 생성
            questions = self.question_gen_agent.generate_questions(
                department=department,
                document=document,
                comment=comment
            )
            # 질문 sort
            ranked_questions = self.priority_agent.generate_priority(
                department=department,
                questions=questions,
            )
            processed_data[id]["qa"]=ranked_questions
            
            # ground_truth 생성
            questions = []
            for qa in processed_data[id]["qa"]:
                ranking = qa.get("ranking")
                category = qa.get("category")
                question = qa.get("question")
                questions.append(f"{ranking}. [{category}]{question}")
            
            if questions != []:
                ground_truth = self.ground_truth_agent.generate_ground_truth(
                    department=department,
                    document=document,
                    questions=questions,
                )
            
            for item in processed_data[id]["qa"]:
                rank = item["ranking"]
                if rank in ground_truth:
                    item["ground_truth"] = ground_truth[rank]
            
            with open(qa_ground_truth_json_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
                
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {fnf_error}")
            traceback.print_exc()  
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error: {json_error}")
            traceback.print_exc()  
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            traceback.print_exc()  
        
        return None