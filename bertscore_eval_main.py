#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.bertscore_eval_pipeline import BERTScoreEvalPipeline

def main():
    parser = argparse.ArgumentParser(description="BERTScore 평가 실행")
    parser.add_argument("--qa_ids", nargs="+", help="평가할 QA ID 목록 (예: 1 2 3)")
    parser.add_argument("--all", action="store_true", help="모든 QA 파일 평가")
    
    args = parser.parse_args()
    
    # BERTScore 평가 pipeline 초기화
    pipeline = BERTScoreEvalPipeline()
    
    try:
        if args.all:
            # 모든 QA 파일 평가
            print("모든 QA 파일에 대해 BERTScore 평가를 시작합니다...")
            results = pipeline.run_evaluation()
        elif args.qa_ids:
            # 특정 QA ID들만 평가
            print(f"QA ID {args.qa_ids}에 대해 BERTScore 평가를 시작합니다...")
            results = pipeline.run_evaluation(args.qa_ids)
        else:
            # 기본적으로 모든 QA 파일 평가
            print("모든 QA 파일에 대해 BERTScore 평가를 시작합니다...")
            results = pipeline.run_evaluation()
        
        if results:
            print(f"\n평가 완료! 총 {len(results)}개의 QA 파일이 평가되었습니다.")
            print("결과는 data/qa/eval/ 디렉토리에 저장되었습니다.")
        else:
            print("평가할 QA 파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


