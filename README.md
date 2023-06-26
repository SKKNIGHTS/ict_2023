## ICT 산업연계 프로젝트

2023 ICT 산업연계 프로젝트 팀 EPIsodic Cognition [EPIC] repository입니다.

`Notion Link` : [https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a](https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a)

## Problem

**ICT 프로젝트가 어떤 문제를 tackle하는지 적어주세요**

## Objective

**ICT 프로젝트의 목표를 적어주세요**

## Dataset

songys님의 "한국어 데이터셋 링크"를 참고했습니다.
[https://github.com/songys/AwesomeKorean_Data](https://github.com/songys/AwesomeKorean_Data)

### Voice Sentiment Dataset



### Text Sentiment Dataset

`감성 대화 말뭉치` : [https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
* 우울증 관련 언어 의미 구조화 및 대화 응답 시나리오 동반한 감성 텍스트 언어 수집
* 크라우드 소싱 수행으로 일반인 1,500명 대상으로한, 음성 10,000 문장 및 코퍼스 27만 문장 구축

![image](https://github.com/a2ran/ict_2023/assets/121621858/78fe94d5-dd35-459f-be89-444315dfe793)

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

```
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-04
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
```

## Services
