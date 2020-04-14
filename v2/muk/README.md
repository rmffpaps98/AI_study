# CycleGAN

# 정의

- GAN을 기반으로 한 이미지 변환 Network. Unpaired된 데이터 셋 A, B를 학습하고 AtoB와 BtoA 이미지 변환을 한다. AtoB 이미지가 Original Image로 돌아갔을 때 이미지의 차이가 없어야 한다.

# 핵심 아이디어

![image](https://user-images.githubusercontent.com/56065194/79143095-93e6ec00-7df7-11ea-8112-92ead1a3d734.png)

- Unpaired Dataset, 즉 X와 Y 도메인의 이미지가 짝을 이루지 않고 각각 하나의 feature를 가진 데이터로 이루어져 있다.

![image](https://user-images.githubusercontent.com/56065194/79143124-a103db00-7df7-11ea-96ad-5f6f460ad8b9.png)

- Cycle Consistency, X -> Y -> X 의 변환을 가능하게 한다. 정반대의 경우도 가능하다.

