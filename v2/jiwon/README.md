# Decision Trees

- 의사결정나무, 결정트리 등으로 불리우며 분류와 회귀 모두에 사용 가능한 모델?입니다.
- 시각적으로 표현 가능하기 때문에 쉽게 이해하고 해석할 수 있습니다.
- 범주형 자료와 수치형 자료 모두에 사용할 수 있습니다.
- 추가 & 다 확인 & 화질

## 구성
![image](https://user-images.githubusercontent.com/61040406/79188468-05578680-7e5a-11ea-8ffc-2122f9627a99.png)
- 화살표의 시작점을 **'parent node(부모 노드)'**, 도착점을 **'child node(자식 노드)'** 라고 부릅니다.
- **root node**: 위 그림에서 파란색 원 부분을 'root node(뿌리 노드)'라고 합니다. 첫 노드로써 모든 데이터가 분류되기 때문에 가장 중요한 노드입니다.
- **internal node**: 노란색 부분은 'internal node(중간 노드)' 혹은 'non-leaf node' 등으로 불리웁니다. 데이터들은 중간 노드를 거쳐가며 노드별 변수의 기준에 따라 분류됩니다.
- **terminal node**: 자식 노드가 없는 파란색 부분은 'terminal node(말단 노드)' 혹은 'leaf node(잎 노드)'라고 합니다. 최종 분류된 데이터의 집합들입니다.


## iris data
- 분류 문제의 기본 예제인 **iris** 데이터를 이용하였습니다.
- 의사결정나무를 이용해 꽃의 종류를 분류하는 것이 목표입니다.
```
from sklearn.datasets import load_iris
iris = load_iris() # X, y = load_iris(return_X_y=True)
print(iris.DESCR) # iris 데이터 설명
```
```
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

==================== 생략...
```
- iris 데이터는 150개의 관측치를 가지며, 4개의 연속형 변수와 1개의 범주형 변수를 가졌.
- 통계학의 아버지로 불리우는 R.A. Fisher가 처음 사용했다고 합니다.


```
# array 형태인 데이터를 보기 편하도록 dataframe으로 변환
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, dtype="category")
y = y.cat.rename_categories(iris.target_names)
df['species'] = y
print(df)
```
![image](https://user-images.githubusercontent.com/61040406/80044285-fac67c80-853e-11ea-92c1-25138aca2fec.png)
- 위 코드는 의사결정나무로 분류하는 데에 꼭 필요하지는 않습니다. array 형태인 iris 데이터를 쉽게 파악할 수 있도록 dataframe 형태로 변환하였습니다.


```
# train set, test set으로 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=1212)
```
- 모델의 학습과 평가를 위해 train set, test set으로 분리하였습니다. 분리 비율은 default 값인 **test_size=0.25(test set)** 입니다.


## Classification
```
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=1212) # criterion='entropy'
clf = clf.fit(x_train, y_train)
```
- train 데이터를 이용하여 트리를 학습시켰습니다. **criterion='gini'** 는 **gini**계수를 기준으로 분기하겠다는 의미입니다. **random_state**는 학습할 때마다 같은 결과를 얻기 위해 사용하였습니다.

```
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, clf.predict(x_train)))
print('{:.3f}'.format(clf.score(x_train, y_train)))
print(confusion_matrix(y_test, clf.predict(x_test)))
print('{:.3f}'.format(clf.score(x_test, y_test)))
# print(clf.predict_proba(x_test))
```
```
[[37  0  0]
 [ 0 38  0]
 [ 0  0 37]]
1.000
[[13  0  0]
 [ 0 10  2]
 [ 0  1 12]]
0.921
```
- confusion matrix를 이용하여 분류 결과를 살펴보았습니다.
- 분류 결과, **train set**에서는 100%의 정확도를 보였고, **test set**에서는 92.1%의 정확도를 보였습니다.
- **train set**의 분류 정확도가 100%인 것으로 보아 **과적합(overfitting)** 의 가능성이 있습니다. 잠시 후 살펴보겠습니다.


## plot_tree
```
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()
```
![image](https://user-images.githubusercontent.com/61040406/80046518-c6a28a00-8545-11ea-9a20-a82612dec0e2.png)
- **train set**으로 학습한 트리를 시각화하였습니다.
- 부모 노드 안에서 가장 윗 줄은 분류 기준을 나타내며, 왼쪽으로 내려가는 화살표는 **'No'** 를, 오른쪽으로 내려가는 화살표는 **'Yes'** 를 의미합니다.
- **gini**는 그 노드에서의 지니계수입니다.
- **sample**은 해당 노드에 들어있는 데이터의 수를 의미하며, 그 아래 **value**는 데이터의 구체적인 분포를 나타냅니다.
- **class**는 해당 노드에 속한 데이터의 분류 결과값입니다.
- 위 트리는 모든 **잎 노드**에서 지니계수가 0이 될 때까지 데이터를 분류하였습니다. 데이터가 하나인 **잎 노드**도 관찰됩니다.
- **train set**에 지나치게 학습된 것으로 보입니다. 이렇게 되면 새로운 데이터(test set 등)가 들어올 때, 과적합으로 인해 제대로 분류하지 못하게 됩니다.


## Pruning
- 트리의 과적합을 막기 위해 **가지치기(pruning)** 를 해야 합니다.
- 말 그대로 트리의 가지를 자르는 것으로 학습 데이터에 대한 분류 정확도는 떨어지겠지만, 새로운 데이터에 대한 정확도를 높일 수 있습니다.
- 가지치기에는 두 가지 방법이 있습니다. **사전 가지치기(pre-pruning)** 과 **사후 가지치기(post-pruning)** 입니다.
- 사전 가지치기는 트리의 **최대 깊이**를 학습 전에 정해주는 것입니다.
```
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1212)
clf = clf.fit(x_train, y_train)
```
- **max_depth=2**를 통해 학습 전에 트리의 최대 깊이를 2로 제한하였습니다.

```
from sklearn.metrics import confusion_matrix
# print(clf.predict_proba(x_test))
print(confusion_matrix(y_train, clf.predict(x_train)))
print('{:.3f}'.format(clf.score(x_train, y_train)))
print(confusion_matrix(y_test, clf.predict(x_test)))
print('{:.3f}'.format(clf.score(x_test, y_test)))
```
```
[[37  0  0]
 [ 0 37  1]
 [ 0  3 34]]
0.964
[[13  0  0]
 [ 0 12  0]
 [ 0  2 11]]
0.947
```
- 그 결과, **train set** 정확도는 100%에서 96.4%로 줄었지만, **test set** 정확도는 92.1%에서 94.7%로 늘었습니다.

```
fig, ax = plt.subplots(figsize=(5, 5))
tree.plot_tree(clf,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()
```
![image](https://user-images.githubusercontent.com/61040406/80047590-c9eb4500-8548-11ea-8a98-1aca25a5fca9.png)


## Feature Importances
```
feature_imp = clf.feature_importances_
plt.barh(iris.feature_names, feature_imp)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
```
![image](https://user-images.githubusercontent.com/61040406/80047941-b2608c00-8549-11ea-8f56-cdd2326dfa57.png)
![image](https://user-images.githubusercontent.com/61040406/80047948-b7254000-8549-11ea-8abf-63bef2def019.png)
- **Feature importance(변수 중요도)** 는 트리가 데이터를 분류하는 데에 있어 각 변수가 어느정도의 영향력를 가지는지를 나타내는 지표입니다.
- **root node**인 'petal length' 변수의 영향력이 가장 컸습니다.


## Algorithm 구현
- **직접 함수를 만들어** 의사결정나무를 학습해보았습니다.
```
def gini(target):
    y_class, class_count = np.unique(target, return_counts=True)
    gini = 1 - sum([(class_count[i] / sum(class_count))**2 
                    for i in range(len(y_class))])
    return np.round(gini, 3)

print('Gini of root node =', gini(y_train))
```
```
Gini of root node = 0.667
```
- **root node**의 지니는 0.667로, sklearn의 의사결정나무와 같습니다.


```
def info_gain(target, split_feature):
    
    # 분기 전 지니
    gini_before = gini(target)
    
    # 분기 후 지니
    gini_after = [0 for i in range(len(split_feature))]
    for i in range(len(split_feature)):
        A = split_feature < split_feature[i]
        B = split_feature >= split_feature[i]
        gini_after[i] = (sum(A) / len(split_feature) * gini(target[A]) + 
                         sum(B) / len(split_feature) * gini(target[B]))
    split_point = split_feature[np.argmin(gini_after)]
    gini_after = np.min(gini_after)
    
    return split_point, np.round(gini_after, 3)

print('split_point, gini =', info_gain(y_train, x_train[:, 0]))
print('split_point, gini =', info_gain(y_train, x_train[:, 1]))
print('split_point, gini =', info_gain(y_train, x_train[:, 2]))
print('split_point, gini =', info_gain(y_train, x_train[:, 3]))
```
```
split_point, gini = (5.5, 0.405)
split_point, gini = (3.4, 0.545)
split_point, gini = (3.3, 0.335)
split_point, gini = (1.0, 0.335)
```
- 첫 분기는 지니계수가 가장 낮은 'petal_length'와 'petal_width' 중 하나를 선택하면 되겠습니다.
- 'petal_length'를 선택할 경우, 역시 sklearn의 의사결정나무와 첫번째 분기점이 같습니다.

```
temp = y_train[x_train[:, 2] < 3.3]
print('Gini =', gini(temp))
```
```
Gini = 0.0
```
- sklearn의 의사결정나무와 같이 'petal_length'가 작은 집합은 해당 노드의 지니가 0이므로 더이상 분기하지 않습니다.

```
temp = y_train[x_train[:, 2] >= 3.3]
print('Gini = ', gini(temp))
print('split_point, gini =', info_gain(temp, x_train[:, 0][x_train[:, 2] >= 3.3]))
print('split_point, gini =', info_gain(temp, x_train[:, 1][x_train[:, 2] >= 3.3]))
# print(info_gain(temp, x_train[:, 2][x_train[:, 2] < 3.3])) # 이미 사용된 변수
print('split_point, gini =', info_gain(temp, x_train[:, 3][x_train[:, 2] >= 3.3])) # petal width로 분기
```
```
Gini =  0.5
split_point, gini = (6.2, 0.413)
split_point, gini = (2.5, 0.442)
split_point, gini = (1.8, 0.1)
```
- sklearn과 같이 'petal_length'가 큰 집합의 지니는 0.5이며, 다음 노드는 지니가 가장 작은(정보 획득이 가장 큰) 'petal_width'를 기준으로 분기합니다.

====================================================================


## Partition
- 의사결정나무에는 다양한 알고리즘이 사용됩니다. 그 중 **ID3**와 **CART** 알고리즘에 대해 알아보겠습니다.
- **ID3**는 인공지능 및 기계학습 분야에서 시작되었고 **C4.5**, **C5.0** 알고리즘으로 발전했고, **CART**는 통계학 분야에서 개발되었습니다.
- **ID3**는 **엔트로피**를 기반으로, **CART**는 **지니**를 기반으로 분기(partition)하며 트리를 만듭니다.

### 엔트로피
![image](https://user-images.githubusercontent.com/61040406/79188850-1fde2f80-7e5b-11ea-8e8c-bb609df5f47b.png)

### 지니
![image](https://user-images.githubusercontent.com/61040406/79188804-06d57e80-7e5b-11ea-8ee0-07446600f3d4.png)


## Pruning
- 과적합
![image](https://user-images.githubusercontent.com/61040406/79183556-bf47f600-7e4c-11ea-87bd-a96007bf0c62.png)

## Reference
[Scikit-Learn](https://scikit-learn.org/stable/modules/tree.html)
