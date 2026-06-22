import ast
from pathlib import Path
import pandas as pd

# "알 수 없음" 보기를 찾기 위한 키워드 (소문자로 비교)
UNKNOWN_KEYWORDS = [
    "cannot be determined",
    "can't be determined",
    "not enough info",
    "not enough information",
    "not known",
    "unknown",
    "undetermined",
    "not answerable",
    "can't answer",
    "cannot answer",
]


def parse_answers(raw: str) -> list[str]:
    """answers 컬럼의 문자열을 실제 리스트로 변환."""
    return ast.literal_eval(raw)


def find_unknown_index(answers: list[str]) -> int | None:
    """선택지 중 '알 수 없음'류 보기의 인덱스를 반환. 없으면 None."""
    for i, ans in enumerate(answers):
        low = ans.lower()
        if any(kw in low for kw in UNKNOWN_KEYWORDS):
            return i
    return None


def load_data(csv_path: str) -> pd.DataFrame:
    """CSV를 읽고, answers 파싱 + unknown 인덱스 컬럼을 추가."""
    df = pd.read_csv(csv_path)
    df["answers"] = df["answers"].apply(parse_answers)
    df["unknown_idx"] = df["answers"].apply(find_unknown_index)
    return df

if __name__ == "__main__":
    df = load_data("data/open/test/test.csv")
    print(df[["sample_id", "answers", "unknown_idx"]].head())
    print("\nunknown_idx 분포:")
    print(df["unknown_idx"].value_counts(dropna=False))