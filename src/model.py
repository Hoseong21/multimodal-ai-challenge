from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model():
    """vLLM 엔진으로 Qwen2.5-VL-7B를 로드. 한 번만 호출."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},  # 프롬프트당 이미지 1장
        dtype="bfloat16",
    )
    return llm


def build_prompt_text(system_prompt: str, user_prompt: str) -> str:
    """Qwen 채팅 템플릿 형식의 단일 프롬프트 문자열 생성 (이미지 자리 포함)."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def infer_one(llm, system_prompt: str, user_prompt: str, image_path: str) -> str:
    """이미지 1장 + 프롬프트로 모델 답변 1개를 생성."""
    image = Image.open(image_path).convert("RGB")
    prompt_text = build_prompt_text(system_prompt, user_prompt)

    sampling = SamplingParams(temperature=0.0, max_tokens=64)

    outputs = llm.generate(
        {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling,
    )
    return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from dataset import load_data
    from prompt import SYSTEM_PROMPT, build_user_prompt

    df = load_data("data/open/test/test.csv")
    row = df.iloc[0]
    user_prompt = build_user_prompt(row["context"], row["question"], row["answers"])
    image_path = f"data/open/test/images/{row['sample_id'].replace('TEST_', 'test_img_')}.jpg"

    print("이미지:", image_path)
    print("로딩 중... (처음엔 모델 다운로드로 몇 분 걸림)")
    llm = load_model()

    output = infer_one(llm, SYSTEM_PROMPT, user_prompt, image_path)
    print("\n=== 모델 출력 ===")
    print(repr(output))
    print("\n=== 정답 선택지 ===")
    print(row["answers"])