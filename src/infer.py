import argparse
import pandas as pd
from tqdm import tqdm

from dataset import load_data
from prompt import SYSTEM_PROMPT, build_user_prompt
from parse import parse_prediction
from model import load_model, infer_one


def sample_id_to_image_path(sample_id: str, image_dir: str) -> str:
    """TEST_0000 -> {image_dir}/test_img_0000.jpg"""
    num = sample_id.replace("TEST_", "")
    return f"{image_dir}/test_img_{num}.jpg"


def run_inference(csv_path, image_dir, output_path, llm, limit=None):
    df = load_data(csv_path)
    if limit is not None:
        df = df.head(limit)

    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = build_user_prompt(
            row["context"], row["question"], row["answers"]
        )
        image_path = sample_id_to_image_path(row["sample_id"], image_dir)
        output = infer_one(llm, SYSTEM_PROMPT, user_prompt, image_path)
        label = parse_prediction(output, row["answers"], row["unknown_idx"])
        preds.append({"sample_id": row["sample_id"], "label": label})
        if limit is not None:  # 스모크 테스트일 때만 상세 출력
            print(f"\n[{row['sample_id']}]")
            print(f"  Q: {row['question']}")
            print(f"  선택지: {row['answers']}")
            print(f"  모델출력: {output!r}")
            print(f"  -> label {label}")

    pd.DataFrame(preds).to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path} ({len(preds)}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/open/test/test.csv")
    parser.add_argument("--image_dir", default="data/open/test/images")
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="처음 N개만 추론 (스모크 테스트용)")
    args = parser.parse_args()

    llm = load_model()
    run_inference(args.csv, args.image_dir, args.out, llm, limit=args.limit)