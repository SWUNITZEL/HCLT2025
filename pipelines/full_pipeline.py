import os
import json
from agents.comment_agent import CommentAgent

class FullPipeline:
    def __init__(self, base_path: str):
        self.comment_agent = CommentAgent()
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
        qa_json_path = os.path.join(self.qa_dir_path, f"qa_{id}.json")
        eval_json_path = os.path.join(self.eval_dir_path, f"eval_{id}.json")
        
        

        # 개별 processed.json 생성
        if not os.path.exists(processed_json_path):
            processed_data={}
            processed_data[id] = {"department":department,
                                  "document":document}
        else:
            with open(processed_json_path, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
        
        # comment가 없을 경우 gpt로 값 생성
        if  not processed_data[id].get("comment"):
            # Comment 생성
            comment = self.comment_agent.generate_comment(
                department=department,
                document=document
            )
            processed_data[id]["comment"]=comment
        
        with open(processed_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        
        return None