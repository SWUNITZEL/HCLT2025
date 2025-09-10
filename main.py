import os
import json
from multiprocessing import Pool, cpu_count
from pipelines.ground_truth_gen_pipeline import GroundTruthGenPipeline

def process_doc(args):
    """단일 문서 처리"""
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

def safe_process(ids_to_process, raw, base_dir, num_processes):
    if not ids_to_process:
        return

    args_list = [(id, raw[id]["department"], raw[id]["document"], base_dir) for id in ids_to_process]

    # 멀티프로세싱
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_doc, args_list)

    # 마지막 ID는 단일 프로세스로 한번 더 안전 처리
    last_id = ids_to_process[-1]
    process_doc((last_id, raw[last_id]["department"], raw[last_id]["document"], base_dir))

    return results

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw.json")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ids = list(raw.keys())

    num_processes = max(cpu_count() - 1, 1)

    # 1차 실행: 전체 문서
    safe_process(ids, raw, BASE_DIR, num_processes)
    print("1차 처리 완료")

    # 누락된 ID 확인
    missing_ids = check_missing_ids(BASE_DIR, ids)
    if missing_ids:
        print("누락된 ground_truth가 있는 ID:", missing_ids)
        # 2차 실행: 누락된 ID만 처리
        safe_process(missing_ids, raw, BASE_DIR, num_processes)
        print("누락된 ID 재처리 완료")
    else:
        print("누락된 ID 없음")
