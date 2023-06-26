## ICT 산업연계 프로젝트

2023 ICT 산업연계 프로젝트 팀 EPIsodic Cognition [EPIC] repository입니다.

`Notion Link` : [https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a](https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a)

## Problem

**ICT 프로젝트가 어떤 문제를 tackle하는지 적어주세요**

## Objective

**ICT 프로젝트의 목표를 적어주세요**

## Dataset

[AwesomeKorean_Data](https://github.com/songys/AwesomeKorean_Data)

[Korean Speech Database for ASR](https://github.com/knlee-voice/AI.Tech/blob/master/docs/KoSpeechDB.md)

### Voice Sentiment Dataset

`감정 분류를 위한 대화 음성 데이터셋` : [https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata)
* 일정 기간동안 사용자들이 어플리케이션과 자연스럽게 대화하고, 수집된 데이터를 정제 작업을 거쳐 선별
* 7가지 감정(happiness, angry, disgust, fear, neutral, sadness, surprise)에 대해 5명이 라벨링

`한국어 음성 감정 데이터셋(KESDy18)` : [https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=eng](https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=eng)
* 헤드셋 마이크(Shure S35) 장치를 통해 수집한 음성데이터에 대한 데이터셋(2018.04~2018.09).
* 한국인 성우 총 30명 (남/여 각 15명) ~ Arousal : (이완) 1-2-3-4-5 (각성) / Valence : (부정) 1-2-3-4-5 (긍정)

### Text Sentiment Dataset

`감성 대화 말뭉치` : [https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
* (크라우드 소싱) 일반인 1,500명 대상 / 음성 15,700문장 / 코퍼스 27만 문장 구축(ALBERT 엔진 학습용)
* 우울증 관련 언어 의미 구조화 및 대화 응답 시나리오 동반 수집 (60가지 감정 상태가 포함, 기본적으로 3턴의 대화를 기준)

![image](https://github.com/a2ran/ict_2023/assets/121621858/78fe94d5-dd35-459f-be89-444315dfe793)

`한국어 감정 정보가 포함된 단발성 대화 데이터셋` : [https://aihub.or.kr/aihubdata/data/view.do?currMenu=118&topMenu=100&dataSetSn=270&aihubDataSe=extrldata](https://aihub.or.kr/aihubdata/data/view.do?currMenu=118&topMenu=100&dataSetSn=270&aihubDataSe=extrldata)
* SNS 글 및 온라인 댓글에 대한 웹 크롤링을 실시하여 문장을 선정함
문장 단위 작업을 수행할 수 있도록 문장 분리 작업을 거침
* 7개 감정(기쁨, 슬픔, 놀람, 분노, 공포, 혐오, 중립) 레이블링 수행

![image](https://github.com/a2ran/ict_2023/assets/121621858/209d5380-ffe7-4ff3-ad03-54ee76b33f5e)

`Multilingual Tweet Intimacy Analysis` : [https://arxiv.org/abs/2210.01108](https://arxiv.org/abs/2210.01108)
* 다국어 트윗 내용, 언어, 그리고 1~5 사이의 실수값을 제공하지만,
* 해당 발화의 정확한 감성 분류 X --> 데이터 사용 부적합

`3i4k` : [https://arxiv.org/pdf/1811.04231.pdf](https://arxiv.org/pdf/1811.04231.pdf)
* 한국어 텍스트 문장을 7가지 의도로 분류함 (Fragment, Statement, Question, Command, Rhetorical Q, Rhetorical C, Into-dep U)
* 동일한 문장이라도 발화자의 의도에 따라 뜻이 달라지니까 멀티모달 모델에 추가 가능할수도!

==> 모델 구축에 있어 `감성 대화 말뭉치` 데이터가 가장 적합. `감성 대화 말뭉치` 데이터 크기는 51,630개로 개수가 많지는 않지만, data augmentation을 통해 노이즈를 넣는 방식으로 데이터 개수를 늘려 학습할 것.

## Guidances

[Huggingface Text Classification Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification)

[Autotrained documentation](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html)

[Text Classification with HuggingFace Transformers Trainer](https://medium.com/grabngoinfo/transfer-learning-for-text-classification-using-hugging-face-transformers-trainer-13407187cf89)

[Finetune DistilBERT for multiclass classification with PyTorch](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb)

## Models

### Voice Sentiment Model

### Text Sentiment Model

**현재 최종 작업중인 모델** : BERT + lightning : [https://drive.google.com/file/d/1T0gOJMMHIPhDndnXeA6cVDNYUzVVYiv-/view?usp=sharing](https://drive.google.com/file/d/1T0gOJMMHIPhDndnXeA6cVDNYUzVVYiv-/view?usp=sharing)

`XGBoost` : [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)

`koBERT` : [https://github.com/monologg/KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)

`감성 대화 말뭉치`은 데이터에 포함된 대화의 주제를 **감정_대분류**, 그리고 **감정_소분류**으로 나누는데, 그 내용은 다음과 같다:

`감정_대분류` : '기쁨' '당황' '분노' '불안' '상처' '슬픔'<br>[감정_대분류] 카테고리 갯수 : 6개

`감정_소분류` : '가난한, 불우한' '감사하는' '걱정스러운' '고립된'...<br>[감정_소분류] 카테고리 갯수 : 58

우리는 감정_대분류, 감정_소분류 두 sentiment를 [감정_대분류 감정 소분류] 형태로 합쳐, 발화자의 세부적인 감정을 분류하고자 한다.

```
train_dataset['sentiment'] = train_dataset[['감정_대분류', '감정_소분류']].apply(lambda x: ' '.join(x), axis=1)
train_dataset = train_dataset.drop(['감정_대분류', '감정_소분류'], axis=1)
```
#### 1. XGBoost

한국어 자연어처리 언어모델인 koBERT (Korean Bidirectional Encoder Representations from Transformers)을 사용해 앞선 데이터의 한국어 문장을 768차원의 숫자 벡터로 임베딩 후, 머신러닝 classification 알고리즘인 XGBoost을 사용해 주어진 instance를 60개의 각기 다른 카테고리로 분류한다.

`XGBoost 작업 notebook 링크` : [text_xgboost.ipynb](https://drive.google.com/file/d/1SEj2o_X4OE2UYOOWJqtzxVbLuzTBzUK5/view?usp=sharing)

```
accuracy = accuracy_score(answer, le.inverse_transform(preds))
print(f'\naccuracy : {accuracy*100:.2f}%')
accuracy : 26.02%
```
모델 분류의 정확도는 26%으로, 아주 낮은 성능을 보인다. 따라서, 머신러닝 알고리즘으로 분류작업을 진행하는 것보다 
딥러닝 프레임워크를 구축해 분류작업을 진행하고자 한다.

#### 2. DistilBERT

koBERT을 사용해 데이터의 한국어 문장을 768차원의 숫자 벡터로 임베딩 후, DistilBERT 딥러닝 프레임워크를 조정해 예측값과 실제값의 loss을 최소화하는 방향으로 모델의 가중치를 업데이트한다.

`DistilBERT 작업 notebook 링크` : [text_DistilBERT.ipynb](https://github.com/a2ran/ict_2023/blob/main/text_sentiment/text_BERTmodel.ipynb)

##### Parameter :

```
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-04
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
```

##### Model Structure :

```
┌────────────────┐
│Input Sequence  │
└───────┬────────┘
        ↓
┌───────┴────────┐
│Embedding Layer │
│(Word + Position│ 
│ + LayerNorm +  │
│   Dropout)     │
└───────┬────────┘
        ↓
┌───────┴────────┐
│TransformerLayer│
│   (ModuleList) │
│      3x        │
│TransformerBlock│
│ (MultiHeadSelf │
│ Attention + FFN│
│ + LayerNorms + │
│    Dropout)    │
└───────┬────────┘
        ↓
┌───────┴───────┐ 
│ PreClassifier │
│   (Linear)    │
└───────┬───────┘
        ↓
┌───────┴───────┐
│  Dropout(0.3) │
└───────┬───────┘
        ↓
┌───────┴────────┐
│  Classifier    │
│   (Linear)     │
│ Out: 60 classes│
└────────────────┘
```

##### Result :

![image](https://github.com/a2ran/ict_2023/assets/121621858/d9286a63-f891-4796-a2b5-8acc9474d220)

![image](https://github.com/a2ran/ict_2023/assets/121621858/e64372bf-2d5b-400a-b431-bf1f6f340106)


##### Evaluation :

for 60 sub-categories :

```
Training Loss Epoch: 0.007094902639563575
Training Data Accuracy = 99.80%
```

```
Validation Loss Epoch: 3.5779860724623385
Validation Data Accuracy = 58.00%
```

for 6 sub-categories :

```
Validation Data Accuracy = 71.33%
```

# **계속 업데이트를 진행해 val_acc > 0.8을 목표로 학습을 진행 예정입니다.**

## Services
