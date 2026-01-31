# YouTube Shorts Maker - 프로젝트 리뷰

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 구조](#아키텍처-구조)
3. [설계 피드백](#설계-피드백)
4. [구조 피드백](#구조-피드백)
5. [개선 제안](#개선-제안)

---

## 프로젝트 개요

### 목적
YouTube Shorts 영상 제작에 필요한 에셋(이미지, 음성)을 AI 에이전트를 통해 자동으로 생성하는 시스템입니다.

### 주요 기능
- **콘텐츠 기획**: 주어진 토픽에서 20초 이내의 Shorts 스크립트 자동 생성
- **이미지 생성**: 각 장면에 맞는 9:16 세로 이미지 생성 (OpenAI GPT-Image-1)
- **음성 생성**: 나레이션 텍스트를 TTS로 변환 (OpenAI TTS)

### 기술 스택
| 구분 | 기술 |
|------|------|
| 에이전트 프레임워크 | Google ADK (Agentic Development Kit) |
| LLM | OpenAI GPT-4o (via LiteLlm) |
| 이미지 생성 | OpenAI GPT-Image-1 |
| 음성 생성 | OpenAI gpt-4o-mini-tts |
| 스키마 검증 | Pydantic |

---

## 아키텍처 구조

### 디렉토리 구조
```
youtube_shorts_maker/
├── agent.py                    # Root Agent (ShortsProducerAgent)
├── prompt.py                   # Root 프롬프트
└── sub_agents/
    ├── prompt.py               # 공유 프롬프트
    ├── content_planner.py      # 콘텐츠 기획 에이전트
    └── asset_generator/
        ├── agent.py            # ParallelAgent (이미지+음성 병렬)
        ├── image_generator/
        │   ├── agent.py        # SequentialAgent
        │   ├── prompt_builder/ # 프롬프트 최적화
        │   └── image_builder/  # 실제 이미지 생성
        └── voice_generator/    # 음성 생성
```

### 에이전트 흐름도
```
[토픽 입력]
     │
     ▼
┌─────────────────────────────────┐
│     ShortsProducerAgent         │  ← Root Agent
│         (GPT-4o)                │
└─────────────────────────────────┘
     │
     ├──────────────────────────────┐
     ▼                              │
┌─────────────────────┐             │
│ ContentPlannerAgent │             │
│   (콘텐츠 기획)      │             │
└─────────────────────┘             │
     │                              │
     │ ContentPlanOutput            │
     │ (scenes, narrations)         │
     ▼                              │
┌─────────────────────────────────┐ │
│   AssetGeneratorAgent           │ │
│      (ParallelAgent)            │ │
│                                 │ │
│  ┌───────────┐  ┌────────────┐  │ │
│  │ Image     │  │  Voice     │  │ │
│  │ Generator │  │ Generator  │  │ │
│  │           │  │            │  │ │
│  │ Sequential│  │            │  │ │
│  │ ┌───────┐ │  │            │  │ │
│  │ │Prompt │ │  │            │  │ │
│  │ │Builder│ │  │            │  │ │
│  │ └───┬───┘ │  │            │  │ │
│  │     ▼     │  │            │  │ │
│  │ ┌───────┐ │  │            │  │ │
│  │ │Image  │ │  │            │  │ │
│  │ │Builder│ │  │            │  │ │
│  │ └───────┘ │  │            │  │ │
│  └───────────┘  └────────────┘  │ │
└─────────────────────────────────┘ │
     │                              │
     ▼                              ▼
[이미지 파일들]              [음성 파일들]
(scene_X_image.jpeg)      (scene_X_narration.mp3)
```

### 에이전트 유형별 역할

| 에이전트 유형 | 사용처 | 특징 |
|-------------|--------|------|
| `Agent` | ContentPlanner, PromptBuilder, ImageBuilder, VoiceGenerator | 기본 LLM + Tools |
| `SequentialAgent` | ImageGenerator | 하위 에이전트 순차 실행 |
| `ParallelAgent` | AssetGenerator | 하위 에이전트 병렬 실행 |

---

## 설계 피드백

### 잘된 점

#### 1. 명확한 관심사 분리 (Separation of Concerns)
각 에이전트가 단일 책임을 가지고 있어 유지보수가 용이합니다.
- ContentPlanner: 기획만 담당
- PromptBuilder: 프롬프트 최적화만 담당
- ImageBuilder: 이미지 생성만 담당

#### 2. 적절한 병렬화 전략
이미지와 음성 생성은 서로 독립적이므로 `ParallelAgent`로 병렬 실행하는 것은 좋은 선택입니다.

#### 3. Pydantic을 통한 타입 안정성
`ContentPlanOutput` 스키마를 정의하여 에이전트 간 데이터 계약을 명확히 했습니다.

#### 4. 캐싱 구현
이미 생성된 파일이 있으면 재생성하지 않는 로직이 tools에 구현되어 있어 효율적입니다.

### 개선이 필요한 점

#### 1. 데이터 전달 방식의 불명확함
현재 `ToolContext`를 통해 이전 에이전트의 결과를 가져오는데, 이 흐름이 암시적입니다.

```python
# 현재 방식 (image_builder/tools.py)
prompt_builder_output = tool_context.state.get("PromptBuilderAgent:output")
```

**문제점**:
- 문자열 키에 의존 → 오타 시 런타임 에러
- 에이전트 이름 변경 시 여러 곳 수정 필요

**제안**:
```python
# 상수 정의
class AgentOutputKeys:
    PROMPT_BUILDER = "PromptBuilderAgent:output"
    CONTENT_PLANNER = "ContentPlannerAgent:output"
```

#### 2. 에러 핸들링 부재
API 호출 실패 시 대응 로직이 없습니다.

```python
# 현재 코드 (tools.py)
response = await client.images.generate(...)  # 실패 시?
```

**제안**:
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_with_retry(client, prompt):
    return await client.images.generate(...)
```

#### 3. 설정값 하드코딩
이미지 크기, 품질 등이 코드에 직접 작성되어 있습니다.

```python
# 현재 방식
size="1024x1536",
quality="low",
```

**제안**: 설정 파일 분리
```python
# config.py
class ImageConfig:
    SIZE = "1024x1536"
    QUALITY = "low"
    FORMAT = "jpeg"
```

#### 4. 출력 경로 관리
파일이 현재 작업 디렉토리에 저장됩니다. 프로젝트별/세션별 관리가 어렵습니다.

**제안**:
```python
# 세션 ID 기반 디렉토리 구조
outputs/
└── {session_id}/
    ├── content_plan.json
    ├── images/
    │   └── scene_1_image.jpeg
    └── audio/
        └── scene_1_narration.mp3
```

---

## 구조 피드백

### 잘된 점

#### 1. 일관된 디렉토리 구조
각 에이전트가 `agent.py`, `prompt.py`, `tools.py` 패턴을 따릅니다.

#### 2. 계층적 구성
Root → Sub-agents → Leaf agents 구조가 명확합니다.

### 개선이 필요한 점

#### 1. 공유 모듈 부재

**현재 문제**:
- `sub_agents/prompt.py`에 일부 공유 프롬프트가 있지만 활용이 불명확
- 공통 유틸리티, 타입, 상수가 없음

**제안된 구조**:
```
youtube_shorts_maker/
├── shared/
│   ├── __init__.py
│   ├── types.py          # 공유 Pydantic 모델
│   ├── constants.py      # 상수 정의
│   ├── config.py         # 설정값
│   └── utils.py          # 공통 유틸리티
├── agent.py
└── sub_agents/
    └── ...
```

#### 2. voice_generator 위치 불일치

**현재**:
```
asset_generator/
├── image_generator/
│   ├── prompt_builder/
│   └── image_builder/
└── voice_generator/      # image_generator와 같은 레벨
```

**불일치점**: `image_generator`는 하위 에이전트가 있는데, `voice_generator`는 단일 레벨입니다.

**제안** (일관성을 위해):
```
asset_generator/
├── image_generator/
│   ├── prompt_builder/
│   └── image_builder/
└── voice_generator/
    └── tts_builder/      # 구조 일관성 (선택적)
```

또는 voice_generator가 단순하다면:
```
asset_generator/
├── image_generator/
│   ├── prompt_builder/
│   └── image_builder/
└── voice_generator.py    # 파일 하나로 통합
```

#### 3. 테스트 구조 부재

**제안**:
```
youtube_shorts_maker/
├── ...
tests/
├── __init__.py
├── conftest.py
├── test_content_planner.py
├── test_image_generator.py
└── test_voice_generator.py
```

#### 4. README.md 내용 부족

현재 README.md가 기본적인 내용만 있습니다. 다음 내용 추가 권장:
- 설치 방법
- 환경 변수 설정 (API 키)
- 사용 예시
- 출력물 설명

---

## 개선 제안

### 우선순위 높음

| 항목 | 이유 | 예상 작업량 |
|------|------|------------|
| 에러 핸들링 추가 | API 호출 실패 시 시스템 안정성 | 중간 |
| 설정 파일 분리 | 유지보수성 향상 | 작음 |
| 출력 경로 관리 | 여러 번 실행 시 파일 충돌 방지 | 작음 |

### 우선순위 중간

| 항목 | 이유 | 예상 작업량 |
|------|------|------------|
| 공유 모듈 생성 | 코드 중복 제거, 타입 안정성 | 중간 |
| 로깅 추가 | 디버깅 및 모니터링 | 작음 |
| README 보강 | 다른 개발자 온보딩 | 작음 |

### 우선순위 낮음

| 항목 | 이유 | 예상 작업량 |
|------|------|------------|
| 테스트 작성 | 회귀 방지 | 큼 |
| voice_generator 구조 일관화 | 코드 일관성 | 작음 |

---

## 코드 예시: 권장 개선사항

### 1. 설정 파일 (shared/config.py)
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    openai_api_key: str

    # Image Generation
    image_size: str = "1024x1536"
    image_quality: str = "low"
    image_format: str = "jpeg"

    # Voice Generation
    default_voice: str = "alloy"
    audio_format: str = "mp3"

    # Output
    output_dir: str = "./outputs"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. 공유 타입 (shared/types.py)
```python
from pydantic import BaseModel
from typing import List

class Scene(BaseModel):
    id: int
    narration: str
    visual_description: str
    embedded_text: str
    embedded_text_location: str
    duration: int

class ContentPlan(BaseModel):
    topic: str
    total_duration: int
    scenes: List[Scene]

class OptimizedPrompt(BaseModel):
    scene_id: int
    enhanced_prompt: str
```

### 3. 에러 핸들링이 추가된 Tools
```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"API 호출 실패, {retry_state.next_action.sleep}초 후 재시도..."
    )
)
async def generate_image_with_retry(client, prompt: str):
    return await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=settings.image_size,
        quality=settings.image_quality,
    )
```

---

## 결론

이 프로젝트는 Google ADK를 활용한 멀티 에이전트 시스템의 좋은 예시입니다. 관심사 분리와 병렬화 전략이 잘 설계되어 있습니다.

다만 프로덕션 레벨로 발전시키려면:
1. **에러 핸들링**과 **재시도 로직** 추가
2. **설정값 외부화**
3. **출력물 관리 체계** 구축
4. **로깅 및 모니터링** 추가

가 필요합니다.

현재 상태로도 프로토타입/MVP로는 충분하며, 점진적으로 위 개선사항을 적용하면 안정적인 시스템으로 발전할 수 있습니다.
