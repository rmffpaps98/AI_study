# AI Study

# GAN(Generative Adversarial Network)

정의
- 적대적 생성 네트워크. Unsupervised Learning의 일종으로, Discriminator와 Generator의 두 모델이 경쟁하며 발전하는 형태의 네트워크이다.

구조

![image](https://user-images.githubusercontent.com/56065194/77943445-2ebfd080-72f8-11ea-9d85-58abab3e5e5a.png)

- Generator는 random noise vector를 받아 fake image를 생성한다. Discriminator는 training set의 real image와 Generator가 생산한 fake image를 판별하여 real과 fake를 구별하고 feedback을 Generator에 전송한다. 그리고 Discriminator는 Ground Truth 즉, training set의 real image로부터 real에 대한 feedback을 받는다.

핵심 아이디어

![image](https://user-images.githubusercontent.com/56065194/77944394-e1446300-72f9-11ea-9e30-c146804d6df9.png)

- 
