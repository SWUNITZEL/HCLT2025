import os
import json
from multiprocessing import Pool, cpu_count
from pipelines.ground_truth_gen_pipeline import GroundTruthGenPipeline

def process_doc(args):
    id, department, document, base_dir = args
    pipeline = GroundTruthGenPipeline(base_dir)
    return pipeline.run(id=id, department=department, document=document)

def check_missing_ids(base_dir, ids):
    """qa_{i}.json 파일을 확인해서 ground_truth가 없는 id 리스트 반환"""
    missing_ids = []
    qa_dir = os.path.join(base_dir, "data", "qa", "ground_truth")
    for i in ids:
        qa_path = os.path.join(qa_dir, f"qa_{i}.json")
        if not os.path.exists(qa_path):
            missing_ids.append(i)
            continue
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            qa_list = data.get(f"{i}", {}).get("qa", [])
            # ground_truth가 3개 이상 있는지 확인
            incomplete = any(
                not (
                    "ground_truth" in item
                    and isinstance(item["ground_truth"], list)
                    and len(item["ground_truth"]) >= 3
                )
                for item in qa_list
            )
            if incomplete:
                missing_ids.append(i)
        except Exception as e:
            print(f"Error checking {qa_path}: {e}")
            missing_ids.append(i)
    return missing_ids

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw.json")
    
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ids = list(raw.keys())

    # 1차 실행
    args_list = [(id, raw[id]["department"], raw[id]["document"], BASE_DIR) for id in ids]
    num_processes = max(cpu_count() - 1, 1)
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_doc, args_list)

    print("모든 문서 처리 완료:", results)

    # 누락된 id 체크
    missing_ids = check_missing_ids(BASE_DIR, ids)
    if missing_ids:
        print("누락된 ground_truth가 있는 ID:", missing_ids)
        # 2차 실행 (누락된 것만)
        args_list = [(id, raw[id]["department"], raw[id]["document"], BASE_DIR) for id in missing_ids]
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_doc, args_list)
        print("누락된 ID 재처리 완료:", results)
    else:
        pass
