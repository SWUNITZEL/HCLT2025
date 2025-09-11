# 파일명: prepare_dataset.py

import os
import json
from typing import List, Dict, Any

def unify_all_qa_sets(base_dir: str, output_dir: str) -> None:
    """
    여러 ground_truth 디렉토리에서 총 300개의 QA 세트를 개별 항목으로 통합하여
    하나의 JSON 파일로 저장합니다.
    """
    print("총 300개 QA 세트 통합을 시작합니다...")
    os.makedirs(output_dir, exist_ok=True)
    
    unified_qa_list = []
    unified_id_counter = 1
    
    # 처리할 디렉토리와 파일 구조 정의
    source_dirs = {
        "ground_truth1": {"prefix": "qa_", "range": range(1, 7), "sets_per_file": 10},
        "ground_truth3": {"prefix": "qa_", "range": range(1, 7), "sets_per_file": 20},
        "ground_truth4": {"prefix": "qa_", "range": range(1, 7), "sets_per_file": 20},
    }

    for dir_name, config in source_dirs.items():
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"경고: '{dir_path}' 디렉토리를 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        for i in config["range"]:
            filename = f"{config['prefix']}{i}.json"
            file_path = os.path.join(dir_path, filename)
            
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data_in_file = json.load(f)
            
            for qa_set_key, qa_set_content in data_in_file.items():
                for qa_pair in qa_set_content.get("qa", []):
                    unified_item = {
                        "unified_id": unified_id_counter,
                        "source_file": filename,
                        "source_dir": dir_name,
                        "department": qa_set_content.get("department", ""),
                        "document": qa_set_content.get("document", ""),
                        "question": qa_pair.get("question", ""),
                        "ground_truths": qa_pair.get("ground_truth", [])
                    }
                    unified_qa_list.append(unified_item)
                    unified_id_counter += 1
    
    # 통합된 데이터 파일로 저장
    unified_file_path = os.path.join(output_dir, "unified_300_qa_sets.json")
    with open(unified_file_path, "w", encoding="utf-8") as f:
        json.dump(unified_qa_list, f, ensure_ascii=False, indent=2)

    print(f"통합 완료! 총 {len(unified_qa_list)}개의 QA 세트를 '{unified_file_path}' 파일에 저장했습니다.")


if __name__ == "__main__":
    BASE_DIRECTORY = "data/qa"
    OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "unified_eval_results_300")
    unify_all_qa_sets(BASE_DIRECTORY, OUTPUT_DIRECTORY)