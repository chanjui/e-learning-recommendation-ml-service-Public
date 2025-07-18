이 프로젝트는 이러닝(e-learning) 플랫폼에서 사용자에게 맞춤형 강의와 강사를 추천하는 머신러닝 기반 백엔드 서비스입니다.  
아래는 데이터가 어떻게 흐르고, 어떤 방식으로 추천이 이루어지는지에 대한 전체적인 설명입니다.

---

## 데이터 파이프라인 및 추천 시스템 전체 흐름

1. **데이터 수집**  
   서비스의 데이터베이스에서 사용자 정보, 강의 정보, 강사 정보, 평점, 수강 이력 등 다양한 데이터를 SQL 쿼리로 추출합니다.  
   추출된 데이터는 Pandas DataFrame으로 변환되어, 이후 데이터 전처리 및 머신러닝 모델 학습에 사용됩니다.

2. **데이터 전처리**  
   - 사용자-강의 평점 데이터는 피벗 테이블 형태의 **사용자-강의 평점 행렬**로 변환됩니다.  
   - 사용자-강사 평점 데이터도 마찬가지로 **사용자-강사 평점 행렬**로 가공됩니다.  
   - 강의의 제목, 설명, 카테고리 등 텍스트 정보는 하나의 문자열로 합쳐져, 텍스트 벡터화에 활용됩니다.  
   - 결측값(평점이 없는 경우 등)은 0으로 대체하여, 머신러닝 모델이 처리할 수 있도록 만듭니다.

3. **머신러닝 기반 추천**  
   - **협업 필터링(Collaborative Filtering)**  
     사용자-강의, 사용자-강사 평점 행렬을 기반으로,  
     scikit-learn의 **NMF(Non-negative Matrix Factorization)** 알고리즘을 사용하여  
     사용자의 잠재적 선호도와 강의/강사의 특성을 추출합니다.  
     이를 통해 각 사용자에게 맞는 강의와 강사를 예측 및 추천합니다.
   - **콘텐츠 기반 추천(Content-Based Filtering)**  
     강의의 텍스트 정보를 **TF-IDF 벡터화**하고,  
     **코사인 유사도**를 계산하여  
     사용자가 수강한 강의와 유사한 강의를 추천합니다.

4. **추천 결과 제공**  
   - 추천 점수 또는 유사도 기반으로 상위 N개의 강의/강사를 선정합니다.  
   - 추천 결과는 API를 통해 프론트엔드 또는 다른 서비스에 전달됩니다.

---

이 파이프라인을 통해,  
**데이터베이스 → 데이터 전처리(Pandas) → 머신러닝 모델(NMF, TF-IDF) → 추천 결과 제공**  
까지의 전 과정을 체계적으로 자동화하였으며,  
모든 민감 정보는 환경변수(.env)로 안전하게 관리합니다.
