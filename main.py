import os
import json
from pipelines.ground_truth_gen_pipeline import GroundTruthGenPipeline
from multiprocessing import Pool, cpu_count

def process_doc(args):
    """
    멀티프로세싱에서 호출할 함수
    """
    id, department, document, base_dir = args
    pipeline = GroundTruthGenPipeline(base_dir)
    print(f"처리 중인 doc ID: {id}")
    pipeline.run(id, department, document)
    return id

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw.json")
    
    # id 가져오기
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)    
    ids = list(raw.keys())
    
    # 멀티프로세싱용 인자 튜플 생성
    args_list = [(id, raw[id]["department"], raw[id]["document"], BASE_DIR) for id in ids]
    
    # CPU 코어 수-1만큼 프로세스 풀 생성
    num_processes = max(cpu_count() - 1, 1)
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_doc, args_list)

    print("모든 문서 처리 완료:", results)
