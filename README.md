# 탄소 친화적 딥러닝 워크로드를 위한 시공간 이동 및 최적화 SW
팀 명 (팀원) : Carbon Watch ([박정현](https://github.com/jhparkland), [김대로](https://github.com/dhfgoeofh))

## 1. 프로젝트 소개

<div align='center'>
  
  ![image](https://github.com/jhparkland/Carbon-friendly/blob/main/intro.png)
  
</div>
인공지능 컴퓨팅 산업의 급격하게 성장하면서 GPU 사용으로 인한 전력 소모량이 큰 폭으로 증가하였습니다.  
본 프로젝트는 딥러닝 모델 학습에 발생하는 방대한 전력소비로 인한 탄소 배출을 줄이기 위해 전력 소모량과 탄소 배출량을 모니터링하고 전력수요 및 전력 안정성에 최적화된 스케줄을 제공하기 위해 만들어졌습니다.   
많은 전력을 필요로 하는 딥러닝 모델 학습의 특성으로 인해 전력 소비를 고려한 DL 학습의 필요성이 부각되고 있습니다.   
국가별 전력생산원의 차이 시간에 따른 재생에너지 발전량 등과 딥러닝의 반복적인 훈련 과정을 이용해 우리는 학습 프로세스를 시공간적으로 이동시킵니다.  
또한, GPU의 주파수를 제안하여 성능에 영향을 크게 미치지 않으며, 탄소 배출을 최소화하는 종합적인 방법을 제안합니다.

## 2. 사용된 기술

<div align='center'>

  ![image](https://github.com/jhparkland/Carbon-friendly/blob/main/migration.png)
  ##### 그림. 2 분산된 클라우드에서 딥러닝 워크로드의 탄소 인지형 이동 기술 

  ![image](https://github.com/jhparkland/Carbon-friendly/blob/main/gpu.png)
  ##### 그림. 3 전력 소비량 최소화를 위한 GPU 주파수 최적화 기술
  
</div>

- **Dash(Plotly)** : Plotly를 기반으로 하는 웹 프레임워크. Dash를 활용하여 Dashboard 제작.
- **Firebase** : Google에서 제공하는 모바일 및 웹 애플리케이션 개발 플랫폼. 실시간 데이터베이스와 같은 다양한 서비스를 제공하여 안전하고, 확장 가능한 애플리케이션 구축.
- **FastAPI** : 웹 프레임워크. 강력한 데이터 검증과 자동 문서화 기능을 제공하여 안정성 강화.
- **nvml** : NVIDIA가 제공하는 NVIDIA GPU 드라이버와 상호 작용하는 라이브러리. GPU의 성능 및 상태 정보에 접근하여 시스템에서 GPU 리소스를 효과적으로 관리. GPU 코어 주파수 측정(nvidia-semi)
- **Moving Between Clouds(MBC)** : 작업을 주기적으로 실행하거나 이벤트에 따라 실행하기 위한 도구. 배치 작업을 자동으로 예약 및 실행하는 데 사용. 훈련 과정의 일부를 시·공간적으로 이동.
- **DVFS** : 동적으로 GPU 전압과 주파수를 조절하여, 성능과 전력 소모를 함께 조절하는 기술.
- **L-BFGS** : 주파수를 최적화하기 위한 알고리즘으로 사용.

## 3. 웹 사이트 이미지

<div align='center'>
  
![image](https://github.com/jhparkland/Carbon-friendly/blob/main/image01.png)
##### 그림. 4 실시간 클라우드의 탄소 배출량 모니터링 SW 메인 패널

</div>

### 성과

- 동아대학교 컴퓨터AI공학부 졸업작품 전시회(FairDay) 최우수 수상 작품. 
- IJCNN 2024 투고완료
