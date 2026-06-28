from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

_processor = None  # 템플릿 생성용 (한 번만 로드)


def load_model():
    """vLLM 엔진으로 Qwen2.5-VL-7B를 로드. 한 번만 호출."""
    global _processor
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
    )
    return llm


def build_prompt_text(system_prompt: str, user_prompt: str) -> str:
    """공식 processor의 채팅 템플릿으로 프롬프트 문자열 생성."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def infer_one(llm, system_prompt: str, user_prompt: str, image_path: str) -> str:
    """이미지 1장 + 프롬프트로 모델 답변 1개를 생성."""
    image = Image.open(image_path).convert("RGB")
    prompt_text = build_prompt_text(system_prompt, user_prompt)

    sampling = SamplingParams(temperature=0.0, max_tokens=64)

    outputs = llm.generate(
        {"prompt": prompt_text, "multi_modal_data": {"image": image}},
        sampling_params=sampling,
    )
    text = outputs[0].outputs[0].text
    return text.strip() if text else ""