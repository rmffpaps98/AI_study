# RANDOM FORESTS(Leo Breiman, 2001)

Θ_k가 뭘까?

## Abstract
- 랜덤 포레스트는 동일한 분포로부터 독립적으로 추출된 랜덤 벡터의 값들에 의존하는 각각의 트리가 모인 조합입니다.
- 모형(forest)의 [일반화 오차(generalization error)](https://ko.wikipedia.org/wiki/%EC%9D%BC%EB%B0%98%ED%99%94_%EC%98%A4%EC%B0%A8)는 트리의 수가 증가할수록 한계(limit)로 수렴합니다.
- 트리 분류기(classifiers) 모형의 일반화 오차는 개별 트리들의 strength(?)와 상관관계에 따라 달라집니다.
- 각 노드를 분할(split)하기 위해 피쳐들을 랜덤 선택하는 것은 [에이다부스트(Adaboost)](https://ko.wikipedia.org/wiki/%EC%97%90%EC%9D%B4%EB%8B%A4%EB%B6%80%EC%8A%A4%ED%8A%B8)와 비교했을 때 더 나은 오분류율(error rate)을 산출하지만, 잡음(noise)에 더욱 강인합니다.
- ??? Internal estimates(?)는 오차, strength, 상관관계를 조절(monitor)하고, 분할에서 사용되는 피쳐들의 수를 증가시키는 반응을 보여주기 위해 사용됩니다. 또한 변수 중요도(variable importance)를 측정하는 데에도 사용됩니다.
- 이 아이디어들은 회귀에도 적용가능합니다.

## 1. Random Forests 
### 1.1. Introduction
- 트리들의 앙상블이 자라고(growing), 그것을 가장 인기있는 클래스에 투표하게 함으로써 분류 정확도는 유의미하게 증가합니다.
- ??? 이 앙상블들을 자라게 하기 위해서, 앙상블 안의 각 트리의 성장을 좌우하는 랜덤 벡터들이 생성됩니다. 배깅(Breiman [1996])과 랜덤 분할 선택(Dietterich [1998])이 그 예입니다.
- ??? 이러한 모든 과정들의 공통점은 k번째 트리에서 동일한 분포에서 추출된 랜덤 벡터들(Θ_1, ..., Θ_(k-1))과 독립인 랜덤 벡터 Θ_k가 생성되고, 트리는 트레이닝 셋과 Θ_k를 이용하고 x가 인풋 벡터인 분류기 h(x,Θ_k)를 만들며 자라납니다.
- 수많은 트리들이 만들어진 후, 다수결 투표를 진행합니다. 이 과정들을 **랜덤 포레스트**라고 합니다.
  - **Definition 1.1**  
  *랜덤 포레스트*는 수많은 트리 기반 분류기들의 집합{h(x,Θ_k), k=1,...}으로 구성된 분류기입니다.   
  Θk들은 i.i.d 랜덤 벡터들이며, 각 트리는 인풋 x에 다수결 투표를 던집니다.

### 1.2. Outline of Paper
- **2. Characterizing the Accuracy of Random Forests**  
이 장에서는 랜덤 포레스트의 몇 가지 이론적 배경을 다룹니다. [강한 큰 수의 법칙(Strong Law of Large Numbers)](https://ko.wikipedia.org/wiki/%ED%81%B0_%EC%88%98%EC%9D%98_%EB%B2%95%EC%B9%99)은 과적합이 문제가 되지 않게 하기 위해, 랜덤 포레스트 모형이 항상 수렴한다는 것을 보입니다.
- **3. Using Random Features** & **4. Random Forests Using Random Input Selection**  
3장에서는 각 노드가 분할을 결정하는 데에 사용될 피쳐들의 랜덤 선택을 소개합니다. 4장에서는 이와 관련된 out-of-bag estimate를 다룹니다.
- **5. Random Forests Using Linear Combinations of Inputs** & **6. Empirical Results on Strength and Correlation**  
 5장과 6장에서는 두 가지 다른 형태의 랜덤 피쳐들에 대해 다룹니다. 첫 번째는 오리지날 인풋으로부터 랜덤 추출을 사용하고, 두 번째는 인풋들의 랜덤 선형 결합을 사용합니다.
- **7. Conjecture: Adaboost is a Random Forest** & **8. The Effects of Output Noise**  
보통 한 개 또는 두 개의 피쳐를 선택하는 것이 최적의 결과를 가져옵니다. 이를 분석하고 이것을 strength, 상관관계와 연결하기 위해 7장에서는 경험적 연구가 이루어집니다. 8장에서는 **에이다부스트**가 **랜덤 포레스트**를 모방했다는 증거를 제시합니다.
- **9. Data with Many Weak Inupts**
- **10. Exploring the Random Forest Mechanism**  
이 장에서는 변수 중요도를 계산하고 이들을 reuse runs(?)와 묶음으로써 **블랙 박스**인 랜덤 포레스트의 메커니즘에 대해 이해해봅니다.
- **11. Random Forests for Regression** & **12. Empirical Results in Regression**  
11장은 회귀를 위한 랜덤 포레스트를 다룹니다. 12장에서는 회귀에 대해 경험적 연구를 수행합니다.
- **13. Remarks and Conclusions**  
13장에서는 결론을 다룹니다.

## 2. Characterizing the Accuracy of Random Forests
### 2.1. Random Forests Converge
- 분류기 h_1(x),h_2(x), ... ,h_K(x)의 앙상블을 가정하면, 랜덤 벡터 Y, X로부터 랜덤하게 추출된 트레이닝 셋과 함께 주변 함수(?)(marginal function)은 다음과 같이 정의합니다.  
mg(X,Y) = av_k I(h_k(X)=Y) − max_(j ≠Y) av_k I(h_k(X)=j) .
