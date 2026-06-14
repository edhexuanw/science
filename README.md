# 초보 학습자를 위한 과학 개념 쉬운 설명 LLM

## 1. 프로젝트 소개

본 프로젝트는 기말 프로젝트의 **10번 시나리오: 도메인 용어 설명가**를 기반으로 진행하였다.

프로젝트의 목표는 과학 분야의 개념이나 전문 용어를 초보 학습자가 이해하기 쉬운 문장으로 설명하는 소형 LLM을 파인튜닝하는 것이다. 단순히 정답만 알려주는 모델이 아니라, 어려운 과학 개념을 짧은 정의와 쉬운 예시로 설명하는 모델을 만드는 데 초점을 두었다.

일반적인 베이스 모델도 과학 개념을 설명할 수 있지만, 답변이 길거나 전문적인 표현이 포함되어 초보자가 이해하기 어려운 경우가 있다. 따라서 본 프로젝트에서는 수업 저장소에서 제공하는 `science_en` 데이터셋을 활용하여, 모델이 과학 개념을 더 간단하고 명확하게 설명하도록 학습시킨다.

주요 대상 사용자는 과학 개념을 처음 배우는 학생, 비전공 대학생, 그리고 영어 과학 용어를 쉬운 문장으로 이해하고 싶은 학습자이다.

---

## 2. 프로젝트 주제

| 항목         | 내용                                  |
| ---------- | ----------------------------------- |
| 선택 시나리오    | 10. 도메인 용어 설명가                      |
| 프로젝트명      | 초보 학습자를 위한 과학 개념 쉬운 설명 LLM          |
| 목표         | 과학 개념을 쉬운 문장과 간단한 예시로 설명하는 LLM 파인튜닝 |
| 사용 데이터     | 수업 저장소에서 제공하는 HuggingFace 데이터셋      |
| 데이터 소스     | `science_en`                        |
| 원본 데이터셋 예시 | `allenai/sciq`                      |

---

## 3. 설치 및 실행 방법

### 3.1 저장소 클론

```bash
git clone https://github.com/xide-projext/edu-llm-colab-unsloth.git
cd edu-llm-colab-unsloth
```

### 3.2 패키지 설치

```bash
pip install -r requirements.txt
pip install datasets
```

### 3.3 학습 데이터 생성

본 프로젝트에서는 수업 저장소에서 제공하는 데이터 생성 스크립트를 사용하였다.
10번 시나리오인 도메인 용어 설명가와 연결하기 위해 과학 교육 데이터셋인 `science_en` 데이터를 선택하였다.

```bash
python scripts/fetch_hf_datasets.py --per-source 8000 --val-ratio 0.05 --only science_en
```

실행 후 다음 파일이 생성된다.

```text
data/hf_train.jsonl
data/hf_val.jsonl
```

데이터는 JSONL 형식으로 구성되며, 기본 구조는 다음과 같다.

```json
{
  "scenario_id": 10,
  "instruction": "Explain the following science concept in simple language.",
  "input": "What is gravity?",
  "output": "Gravity is a force that pulls things down toward Earth."
}
```

### 3.4 Colab에서 학습 실행

1. Google Colab에서 `notebooks/unsloth_edu_finetune.ipynb` 파일을 연다.
2. 런타임 유형을 **T4 GPU**로 설정한다.
3. 베이스 모델을 다음과 같이 설정한다.

```python
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
```

4. 학습 데이터 경로를 확인한다.

```python
TRAIN_PATH = "data/hf_train.jsonl"
VAL_PATH = "data/hf_val.jsonl"
```

5. 모든 셀을 순서대로 실행하여 파인튜닝을 진행한다.

---

## 4. 주요 기능

파인튜닝된 모델은 다음과 같은 기능을 목표로 한다.

* 과학 용어를 쉬운 문장으로 설명한다.
* 어려운 개념을 짧고 명확하게 정리한다.
* 초보 학습자가 이해하기 쉬운 표현을 사용한다.
* 가능하면 쉬운 예시를 함께 제공한다.
* 베이스 모델보다 더 일관된 설명 형식을 제공한다.

### 간단한 예시

입력:

```text
What is gravity?
```

예상 출력:

```text
Gravity is a force that pulls things down toward Earth. For example, when you drop a ball, it falls because of gravity.
```

입력:

```text
What is light?
```

예상 출력:

```text
Light is energy that helps us see things. The Sun and lamps give us light.
```

입력:

```text
What is heat?
```

예상 출력:

```text
Heat is warm energy. It can make ice melt or water become hot.
```

---

## 5. 사용 기술 스택

| 구분         | 사용 기술                           |
| ---------- | ------------------------------- |
| 실행 환경      | Google Colab / T4 GPU           |
| 베이스 모델     | `unsloth/Qwen2.5-0.5B-Instruct` |
| 파인튜닝 프레임워크 | Unsloth                         |
| 학습 방식      | SFT, LoRA / QLoRA               |
| 데이터 처리     | HuggingFace Datasets, JSONL     |
| 학습 라이브러리   | TRL, PEFT, bitsandbytes         |
| 모델 비교      | Base Model vs Fine-tuned Model  |
| 실행 방법      | Colab 추론 또는 GGUF export 후 로컬 실행 |

---

## 6. 파인튜닝 설계 설명

### 6.1 베이스 모델 선택 이유

본 프로젝트에서는 `unsloth/Qwen2.5-0.5B-Instruct` 모델을 베이스 모델로 선택하였다.

이 모델을 선택한 이유는 다음과 같다.

* 모델 크기가 작아 Google Colab T4 GPU 환경에서 학습하기 적합하다.
* Instruct 모델이기 때문에 질문과 답변 형식의 데이터 학습에 적합하다.
* Unsloth에서 지원되어 LoRA/QLoRA 기반 파인튜닝을 비교적 빠르게 진행할 수 있다.
* 기본적인 영어 이해 능력을 가지고 있어 `science_en` 데이터셋 학습에 적합하다.

### 6.2 데이터셋 선택 이유

본 프로젝트의 주제는 과학 도메인 용어 설명가이므로, 과학 개념과 관련된 데이터가 필요하다.
따라서 수업 저장소에서 제공하는 HuggingFace 데이터셋 중 `science_en` 소스를 사용하였다.

이 데이터셋은 과학 질문과 답변 형태를 포함하고 있어, 모델이 과학 개념을 설명하는 방식과 답변 구조를 학습하는 데 적합하다.

### 6.3 학습 방식 선택 이유

전체 모델을 다시 학습하는 것은 많은 GPU 메모리와 시간이 필요하다.
따라서 본 프로젝트에서는 LoRA/QLoRA 방식을 사용하였다.

LoRA는 모델 전체를 수정하지 않고 일부 어댑터 파라미터만 학습하는 방식이다. 그래서 작은 GPU 환경에서도 효율적으로 파인튜닝할 수 있다. 또한 QLoRA는 모델을 더 적은 메모리로 학습할 수 있게 해 주기 때문에 Colab 환경에 적합하다.

### 6.4 하이퍼파라미터 설정

| 항목                    |  설정값 | 선택 이유                        |
| --------------------- | ---: | ---------------------------- |
| LoRA rank             |   16 | 작은 모델에서 충분한 표현력을 확보하기 위해 선택  |
| LoRA alpha            |   16 | rank와 균형을 맞추기 위해 설정          |
| LoRA dropout          | 0.05 | 과적합을 줄이기 위해 사용               |
| Learning rate         | 2e-4 | LoRA 파인튜닝에서 자주 사용하는 안정적인 학습률 |
| Epoch                 |    1 | Colab 환경에서 빠르게 실험하기 위해 설정    |
| Batch size            |    2 | GPU 메모리 사용량을 줄이기 위해 설정       |
| Gradient accumulation |    4 | 작은 batch size를 보완하기 위해 사용    |
| Max sequence length   | 1024 | 짧은 과학 설명 데이터에 적합한 길이로 설정     |

---

## 7. Base Model vs Fine-tuned Model 비교

동일한 질문을 베이스 모델과 파인튜닝 모델에 입력하여 답변 차이를 비교하였다.
아래 표는 학습 후 실제 결과를 바탕으로 수정할 예정이다.

| 입력               | Base Model                                    | Fine-tuned Model                                                                                              | 개선점                              |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| What is gravity? | 중력에 대해 설명하지만 문장이 길고 초보자가 바로 이해하기 어려운 부분이 있었다. | Gravity is a force that pulls things down toward Earth. For example, a dropped ball falls because of gravity. | 더 짧고 쉬운 문장으로 설명하고 간단한 예시를 포함하였다. |
| What is light?   | 빛에 대한 과학적 설명은 가능하지만 표현이 다소 추상적이었다.            | Light is energy that helps us see things. The Sun and lamps give us light.                                    | 초보 학습자가 이해하기 쉬운 일상적인 표현을 사용하였다.  |
| What is heat?    | 열에 대해 설명하지만 핵심 개념이 길게 표현되었다.                  | Heat is warm energy. It can make ice melt or water become hot.                                                | 짧은 정의와 쉬운 예시를 함께 제공하였다.          |

---

## 8. 학습 결과

학습 과정에서는 training loss를 확인하여 모델이 실제로 학습되고 있는지 확인한다.

| 항목               | 결과                              |
| ---------------- | ------------------------------- |
| 사용 모델            | `unsloth/Qwen2.5-0.5B-Instruct` |
| 사용 데이터           | `science_en`                    |
| 학습 방식            | LoRA / QLoRA                    |
| 학습 환경            | Google Colab T4 GPU             |
| 최종 training loss | 학습 후 입력 예정                      |
| 검증 결과            | 학습 후 입력 예정                      |

---

## 9. 실행 화면 및 스크린샷

학습 완료 후 아래 위치에 스크린샷을 추가할 예정이다.

| 항목         | 파일 경로                          |
| ---------- | ------------------------------ |
| 학습 로그      | `images/training_log.png`      |
| Loss Curve | `images/loss_curve.png`        |
| 모델 비교      | `images/model_comparison.png`  |
| 추론 예시      | `images/inference_example.png` |

### 9.1 학습 로그

![training log](images/training_log.png)

### 9.2 Loss Curve

![loss curve](images/loss_curve.png)

### 9.3 Base Model과 Fine-tuned Model 비교

![model comparison](images/model_comparison.png)

### 9.4 추론 예시

![inference example](images/inference_example.png)

---

## 10. 결론

본 프로젝트에서는 10번 시나리오인 도메인 용어 설명가를 기반으로, 과학 개념을 쉬운 문장으로 설명하는 LLM을 파인튜닝하였다.

수업 저장소에서 제공하는 `science_en` 데이터셋을 사용하였고, `unsloth/Qwen2.5-0.5B-Instruct` 모델에 LoRA/QLoRA 방식으로 학습을 진행하였다.

파인튜닝 후 모델은 베이스 모델보다 더 짧고 명확한 답변을 생성하는 것을 목표로 한다. 특히 과학 개념을 초보 학습자가 이해하기 쉬운 표현과 간단한 예시로 바꾸는 데 초점을 두었다.

향후에는 한국어 과학 용어 데이터셋을 추가하여 한국어 설명 능력을 강화하고, 더 다양한 과학 분야의 용어 설명으로 확장할 수 있다.

---

## 11. 참고 자료

* Unsloth Studio
* Unsloth Fine-tuning Guide
* LoRA: Hu et al., 2021
* QLoRA: Detmers et al., 2023
* HuggingFace Datasets
* `allenai/sciq`
* 수업 저장소: `edu-llm-colab-unsloth`
