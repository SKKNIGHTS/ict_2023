## ICT 산업연계 프로젝트

2023 ICT 산업연계 프로젝트 팀 EPIsodic Cognition [EPIC] repository입니다.

`Notion Link` : [https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a](https://www.notion.so/ICT-bdd82a5b00f6426f941aeaf569a3815a)

## Problem

## Objective

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

`koBERT` : [https://huggingface.co/monologg/kobert](https://huggingface.co/monologg/kobert)

## Services
