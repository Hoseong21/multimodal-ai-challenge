import argparse
import pandas as pd
from tqdm import tqdm

from dataset import load_data
from prompt import SYSTEM_PROMPT, build_user_prompt
from parse import parse_prediction


def mock_model(system_prompt: str, user_prompt: str, answers: list[str]) -> str:
    """진짜 모델 대신 쓰는 가짜. 일단 마지막 선택지를 그대로 반환."""
    return answers[-1]


def run_inference(csv_path: str, image_dir: str, output_path: str, model_fn):
    df = load_data(csv_path)
    preds = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = build_user_prompt(
            row["context"], row["question"], row["answers"]
        )
        # 실제로는 여기서 이미지도 같이 넘기지만, mock 단계에선 텍스트만
        output = model_fn(SYSTEM_PROMPT, user_prompt, row["answers"])
        label = parse_prediction(output, row["answers"], row["unknown_idx"])
        preds.append({"sample_id": row["sample_id"], "label": label})

    pd.DataFrame(preds).to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path} ({len(preds)}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/open/test/test.csv")
    parser.add_argument("--image_dir", default="data/open/test/images")
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args()

    run_inference(args.csv, args.image_dir, args.out, mock_model)