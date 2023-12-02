# 2023 바이오헬스 데이터 경진대회 - 치의학 분야 (바이오헬스 혁신융합대학 7개 대학 재학생 부문)
### 1st prize
## [컴퓨터비전] 사랑니 발치 수술 후 위험도 예측 모델 개발
![image](https://github.com/morateng/BioHealth_Data_AI_Competition/assets/100129662/9b4e8f23-8a80-41cd-b4b5-656e6b677327)

### 실행방법

1. 사용할 모델들 학습
    
    ```
    # /DATA/train/images 경로 상의 이미지들과 /DATA/train/train.csv 사용
    # 실행결과 : ./save/ResNet/train_best.pth 저장
    python train_resnet.py
    # 실행결과 : ./save/EfficientNet/train_best.pth 저장
    python train_efficientnet.py
    # 실행결과 : ./save/DenseNet/train_best.pth 저장
    python train_densenet.py
    ```
    
2. 1번으로 학습된 모델의 가중치를 활용한 추가 학습
    
    ```
    # /DATA/train/images 경로 상의 이미지들과 /DATA/train/train.csv 사용
    # 실행결과 : ./save/ResNet/train_best_extend.pth 저장
    python train_resnet_extend.py
    # 실행결과 : ./save/EfficientNet/train_best_extend.pth 저장
    python train_efficientnet_extend.py
    # 실행결과 : ./save/DenseNet/train_best_extend.pth 저장
    python train_densenet_extend.py
    ```
    
3. 앙상블 예측
    
    ```
    # /DATA/test/images 경로 상의 이미지들과 /DATA/test/test.csv 사용
    # 세 모델의 train_best_extend.pth 가중치 활용
    # 추가 학습한 모델들로 앙상블 예측
    # 실행결과 : ./submission.csv 저장됨
    python predict_ensemble.py
    ```
    

### 환경

1. 폴더 구조
    
    ```
    /USER/RESULT
    ├── save/
    │   └── ResNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    │   └── EfficientNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    │   └── DenseNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    ├── models/
    │   ├── resnet50_weights.pth
    │   ├── densenet121_weights.pth
    │   └── efficientnet_v2_s_weights.pth
    ├── utils/
    │   ├── torchinfo/
    │   ├── augmentation.py
    │   ├── preprocessing.py
    │   ├── check.py
    │   ├── image_mean.py
    │   └── download_weights.py
    ├── weight_on_submission/
    │   └── ResNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    │   └── EfficientNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    │   └── DenseNet/
    │       ├── train_best.pth
    │       └── train_best_extend.pth
    ├── README.md
    ├── train_resnet.py
    ├── train_resnet_extend.py
    ├── train_efficientnet.py
    ├── train_efficientnet_extend.py
    ├── train_densenet.py
    ├── train_densenet_extend.py
    ├── predict_ensemble.py
    ├── final_submission.csv
    └── submission.csv
    ```
    
    - train_resnet.py : ResNet50 학습 후 가중치 저장하는 코드
    - train_efficient.py : EfficientNetv2_s 학습 후 가중치 저장하는 코드
    - train_densenet.py : DenseNet121 학습 후 가중치 저장하는 코드
    - train_resnet_extend.py : ResNet50 추가 학습 후 가중치 저장하는 코드
    - train_efficient_extend.py : EfficientNetv2_s 추가 학습 후 가중치 저장하는 코드
    - train_densenet_extend.py : DenseNet121 추가 학습 후 가중치 저장하는 코드
    - weight_on_submission : 최종 결과 제출 시 사용한 가중치들
    - final_submission.csv : 최종 결과 제출한 csv
    - utils : 데이터셋 클래스와 증강, 전처리, eda, 가중치 다운로드 등과 같이 보조 코드들
    - models : 사전학습 가중치들
    
2. 환경 및 라이브러리
    
    T4 GPU (16GB), CPU 10core 96GB MEM GPU
    
    python : 3.9.18
    • cuda : runtimeAPI 11.3, driverAPI 11.7
    • os : Ubuntu 20.04 LTS
    
    라이브러리
    
    ```
    albumentations==1.3.1
    anyio==3.6.2
    argon2-cffi==21.3.0
    argon2-cffi-bindings==21.2.0
    arrow==1.2.3
    asttokens==2.2.1
    attrs==22.1.0
    Babel==2.11.0
    backcall==0.2.0
    beautifulsoup4==4.11.1
    bleach==5.0.1
    certifi==2019.11.28
    cffi==1.15.1
    chardet==3.0.4
    comm==0.1.1
    dbus-python==1.2.16
    debugpy==1.6.4
    decorator==5.1.1
    defusedxml==0.7.1
    distro==1.4.0
    distro-info===0.23ubuntu1
    entrypoints==0.4
    executing==1.2.0
    fastjsonschema==2.16.2
    filelock==3.9.0
    fqdn==1.5.1
    fsspec==2023.4.0
    idna==2.8
    imageio==2.33.0
    importlib-metadata==5.1.0
    ipykernel==6.19.0
    ipython==8.7.0
    ipython-genutils==0.2.0
    isoduration==20.11.0
    jedi==0.18.2
    Jinja2==3.1.2
    joblib==1.3.2
    json5==0.9.10
    jsonpointer==2.3
    jsonschema==4.17.3
    jupyter-events==0.5.0
    jupyter_client==7.4.8
    jupyter_core==5.1.0
    jupyter_server==2.0.0
    jupyter_server_terminals==0.4.2
    jupyterlab==3.5.1
    jupyterlab-pygments==0.2.2
    jupyterlab_server==2.16.3
    lazy_loader==0.3
    lion-pytorch==0.1.2
    MarkupSafe==2.1.1
    matplotlib-inline==0.1.6
    mistune==2.0.4
    mpmath==1.3.0
    nbclassic==0.4.8
    nbclient==0.7.2
    nbconvert==7.2.6
    nbformat==5.7.0
    nest-asyncio==1.5.6
    networkx==3.0
    notebook==6.5.2
    notebook_shim==0.2.2
    numpy==1.24.1
    opencv-python==4.8.1.78
    opencv-python-headless==4.8.1.78
    packaging==22.0
    pandas==2.1.3
    pandocfilters==1.5.0
    parso==0.8.3
    pexpect==4.8.0
    pickleshare==0.7.5
    Pillow==9.3.0
    platformdirs==2.6.0
    prometheus-client==0.15.0
    prompt-toolkit==3.0.36
    psutil==5.9.4
    ptyprocess==0.7.0
    pure-eval==0.2.2
    pycparser==2.21
    Pygments==2.13.0
    PyGObject==3.36.0
    pyrsistent==0.19.2
    python-apt==2.0.0+ubuntu0.20.4.8
    python-dateutil==2.8.2
    python-json-logger==2.0.4
    pytz==2022.6
    PyYAML==6.0
    pyzmq==24.0.1
    qudida==0.0.4
    requests==2.22.0
    requests-unixsocket==0.2.0
    rfc3339-validator==0.1.4
    rfc3986-validator==0.1.1
    scikit-image==0.22.0
    scikit-learn==1.3.2
    scipy==1.11.4
    Send2Trash==1.8.0
    six==1.14.0
    sniffio==1.3.0
    soupsieve==2.3.2.post1
    ssh-import-id==5.10
    stack-data==0.6.2
    sympy==1.12
    terminado==0.17.1
    threadpoolctl==3.2.0
    tifffile==2023.9.26
    tinycss2==1.2.1
    tomli==2.0.1
    torch==2.1.1+cu118
    torchaudio==2.1.1+cu118
    torchvision==0.16.1+cu118
    tornado==6.2
    tqdm==4.66.1
    traitlets==5.6.0
    triton==2.1.0
    typing_extensions==4.4.0
    tzdata==2023.3
    unattended-upgrades==0.1
    uri-template==1.2.0
    urllib3==1.25.8
    wcwidth==0.2.5
    webcolors==1.12
    webencodings==0.5.1
    websocket-client==1.4.2
    wheel==0.38.4
    zipp==3.11.0z
    ```
    
3. 사용 모델 및 가중치
    1. [https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s)
    2. [https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html?highlight=resnet#torchvision.models.resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html?highlight=resnet#torchvision.models.resnet50)
    3. [https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html?highlight=densenet#torchvision.models.densenet121](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html?highlight=densenet#torchvision.models.densenet121)
    
    해당 모델들의 *IMAGENET1K_V1 사전학습 가중치 사용*
    

### 전처리

1. 학습 데이터 라벨 분포 살펴보기
2. 정규화 수치 결정
    
    학습 데이터 이미지 화소값들을 /USER/RESUTL/utils/image_mean.py를 통해 평균과 표준편차 확인 후 설정
    
    평균 : 0.52, 표준편차 : 0.11
    
3. 데이터 증강
    1. 원본 이미지
    2. 수평 flip 이미지
    3. randomcrop and rotation된 이미지
    
    세 이미지를 concat하여 학습에 활용
    

### 학습

1. 사용한 모델
    1. ResNet
    2. EfficientNet
    3. DenseNet
2. 옵티마이저 = Lion
    
    [https://github.com/lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch)
    
    - 배치사이즈 = 170
    
    → 해당 옵티마이저 논문에서 배치가 클수록 성능이 좋아진다는 것을 확인 후 현재 코드에서 cuda 메모리가 허용하는 가장 큰 배치사이즈로 결정
    
    - 학습률 = 1e-4
    - 가중치 감쇠 = 1
    
    → 학습률과 가중치 감쇠는 validation 데이터를 생성한 실험에서 validation score가장 크고 train loss가 가장 작아지는 값으로 휴리스틱하게 결정
    
3. 스케줄러 = LambdaLR (*lr_lambda*=lambda *epoch*: 0.99 ** *epoch*, *last_epoch*=-1)
    
    → 스케줄러의 파라미터는 validation 데이터를 생성한 실험에서 validation score가장 크고 train loss가 가장 작아지는 값으로 휴리스틱하게 결정
    
4. 손실함수 = BCEWithLogitsLoss
    
    50회 학습 후와 10회 추가 학습 후의 로스값 변화
    
    Resent 0.1330 -> 0.1181
    
    Efficientnet 0.1530  -> 0.1246
    
    Densenet  0.1508 -> 0.1277
    
- **학습 순서**
    1. 1차 학습: 모든 모델들을 50 에포크씩 학습
    2. 추가 학습: 1차 학습 결과로 나온 가중치로 모든 모델들을 10 에포크씩 추가 학습
    3. 앙상블: 추가 학습으로 나온 가중치를 가지고 모델들의 output을 가중합으로 앙상블 진행
    - 1차 학습과 2차 학습에 사용한 데이터는 validation 데이터 없이 제공된 모든 train 데이터를 사용.
        
         → train과 validation 분할후 validation 데이터와 함께 학습한 실험에서 train 데이터로만 학습했을 경우 validation의 loss가 점점 증가하는 경향을 관찰하였으며 train 데이터로만 학습했던 가중치로 validation 데이터만 추가 학습 시켰을 경우 train 데이터에 대한 score가 점점 감소하는 경향을 관찰할 수 있었음. 이에 train과 validation의 데이터 분포가 많이 다르다고 판단하여 전체 데이터로 학습하기로 결정함.
        
        → 1차 학습의 경우 학습 중 많은 에포크가 지나면 스케줄러의 영향으로 train loss가 줄지 않거나 변동하는 현상을 관찰할 수 있었는데 스케줄러의 영향으로 판단해 이를 극복하기 위해 1차학습과 동일한 설정으로 가중치만 가져와 추가 학습시킴.
        
    - 모든 데이터는 위 **전처리** 부분에 기술된 데이터 증강을 적용

### 예측

1. 앙상블 학습
    1. 가중치 설정
        
        outputs_res * 0.6 + outputs_effi * 0.2 + outputs_den * 0.2
        
        → 앙상블 가중치는 validation 데이터를 사용한 실험에서 제일 score가 높았던 모델인 resnet에게 0.6을 주고 나머지 모델들은 동일하게 0.2씩 적용
        

### About train_(resnet, efficientnet, densenet).py # 세 파일에 관한 설명

- train data에 대한 resnet, efficientnet, densenet 모델의 학습을 하는 python 코드
- **class ResNet(nn.Module)**
    - 사용 모델을 정의
    - pytorch의 nn.Module을 상속받음
    - pretrained resnet50 모델을 사용
    - 모델의 fc(fully connected) layer을 다음과 같이 설정
    
    ```
    self.fc = nn.Sequential(
                nn.Linear(n_features, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)
            )
    ```
    
    - forward() 함수를 통해 output 출력
- **class EfficientNet(nn.Module)**
    - pretrained efficientnet_v2_s 모델을 사용
    - 모델의 분류기 구조를 class ResNet과 동일한 fc layer 구조 사용
- **class DenseNet(nn.Module)**
    - pretrained densenet121 모델을 사용
    - fc layer 구조와 forward 함수는 class ResNet과 동일
- **def train(model, train_loader, criterion, optimizer, scheduler, device)**
    - train_loader를 통해 train data의 이미지와 라벨값을 받음
    - 데이터를 모델의 input으로 넣은 output과 label의 loss를 계산해 backpropagation과 optimizer을 통한 학습을 수행
    - 학습 중 발생한 loss를 리턴
- **Data Augumentation**
    - 원본이미지, flip 이미지, randomcrop and rotation된 이미지를 모두 concat해주어 데이터 증강
    - DataLoader()로 감싸 배치처리 할 수 있게 함
- **for epoch in range(num_epochs)**
    - 위에서 설정한 epoch수 만큼 모델의 학습을 진행
    - train loss의 추세를 보며 loss가 이전 평균loss보다 작으면 모델의 가중치를 저장
    

### About train_(resnet, efficientnet, densenet)_extend.py # 세 파일에 관한 설명

- train_(resnet, efficientnet, densenet).py 에서 학습한 모델들의 가중치를 받아와서 재학습하는 코드
- model.load_state_dict() 부분을 train_(resnet, efficientnet, densenet).py 를 통해 나온 가중치로 설정
- 그 외의 코드는 train_(resnet, efficientnet, densenet).py와 동일

### About predict_ensemble.py

- 테스트 데이터의 예측을 하는 python 코드
- **사용 모델 class 정의**
    - 앙상블에 사용할 세 모델(ResNet, EfficientNet, DenseNet)을 학습할 때와 동일한 구조를 class로 정의
- **Transform**
    - test 이미지를 224x224로 Resize하고 정규화하는 코드
- **모델 선언**
    - 예측에 사용할 모델들(ResNet, EfficientNet, DensNet)을 선언하고 가중치를 최종 학습한 모델의 가중치로 설정함
- **for image in tqdm(test_loader)**
    - 세 모델의 test data에 대한 예측값을 받음(outputs_res, output_den, output_effi)
    - 모델 앙상블을 위해 outputs_ensemble에 각 모델의 예측값 * 가중치를 다 더해서 할당함
        - 0.6 * ResNet예측값 + 0.2 * EfficientNet예측값 + 0.2 * DenseNet예측값
    - outputs_ensemble에 시그모이드를 취하고 반올림 한 값을 최종 test data에 대한 class로 판별한다(1 == ‘high’, 0 == ‘low’)

### 사용한 기타 .py 설명

- 기타 기능을 위한 파일들을 utils라는 패키지에 모아 사용
- check.py : 학습 데이터 타겟 라벨의 불균형 여부를 확인하기 위한 파일
- download_weights.py : 사전학습 모델들의 가중치 다운로드하기 위한 파일
- preprocessing.py : 다양한 전처리 시도와 데이터셋 클래스를 위한 파일
    - **class BasicTrainDataset(Dataset)**
        - train.csv 에서 파일 이름을 불러와 transform 진행하여 이미지와 라벨을 반환하는 데이터셋 클래스
    - **class CombinedTrainDataset(Dataset)**
        - 같은 이미지이지만 처리 방법이 다른 각각의 1채널 이미지 데이터셋을 3채널 이미지로 변환하여 반환하는 데이터셋 클래스
    - **class CombinedTrainDataset2(Dataset)**
        - 같은 이미지이지만 처리 방법이 다른 각각의 1채널 이미지 데이터셋을 2채널 이미지로 변환하여 반환하는 데이터셋 클래스
    - **class BasicPredictDataset(Dataset)**
        - test.csv에서 파일 이름들을 불러와 transform 진행하여 이미지로 반환하는 데이터셋 클래스
    - **class CombinedPredictDataset(Dataset)**
        - 같은 테스트 이미지이지만 처리 방법이 다른 각각의 1채널 테스트 이미지 데이터셋를 3채널 이미지로 변환하여 반환하는 데이터셋 클래스
    - **class OpenTransform**
        - 그레이 스케일 된 1채널 이미지 텐서를 입력으로 받아 모폴로지 열림 연산 이후 255로 나누는 정규화 진행하여 반환하는 변환 연산 클래스
    - **class HistTransform**
        - 그레이 스케일 된 1채널 이미지 텐서를 입력으로 받아 히스토그램 평활화 이후 255로 나누는 정규화 진행하여 반환하는 변환 연산 클래스
    - **class CannyTransform**
        - 그레이 스케일 된 1채널 이미지 텐서를 입력으로 받아 low threshold가 50, high threshold가 150으로 설정된 캐니 엣지 탐지 연산 진행 이후 255로 나누는 정규화 진행하여 반환하는 변환 연산 클래스
    - **class DBSCANTransform**
        - 그레이 스케일 된 1채널 이미지 텐서를 입력으로 받아 밝기값의 1차원 배열로 변환하여 eps가 0.5 min_samples가 5인 군집화하고, 군집에 따른 색상값을 지닌 이미지로 변환한 이후 가장 큰 군집 라벨 값으로 나누는 정규화 진행하여 반환하는 변환 연산 클래스
    - **class HDBSCANTransform**
        - 그레이 스케일 된 1채널 이미지 텐서를 입력으로 받아 밝기값의 1차원 배열로 변환하여 HDBSCAN으로 밝기값에 대한 군집화하고, 군집에 따른 색상값을 지닌 이미지로 변환한 이후 가장 큰 군집 라벨 값으로 나누는 정규화 진행하여 반환하는 변환 연산 클래스
    - **def getTransformations()**
        - 여러 변환연산 시퀀스를 사전에 정의해두고 반환해주는 함수로, 학습이나 예측코드에서 호출해서 사용
- augmentation.py : 다양한 데이터 증강 시도와 증강된 데이터셋 클래스를 위한 파일
    - **class AugmentationDataset(Dataset)**
        - 증강할 데이터셋을 입력으로 받아 변환을 적용하여 변환된 이미지와 원래 라벨을 반환해주는 데이터셋 클래스
    - **def get_Aug_Transformations()**
        - 데이터 증강을 위한 변환 연산 시퀀스를 사전에 정의해두로 이미지 증강이 필요할 때 호출해서 사용
- imagemean.py
    - 3채널 train 이미지의 각 채널마다의 화소값의 평균과 표준편차를 확인하기 위해 사용
- torchinfo
    - [https://github.com/TylerYep/torchinfo](https://github.com/TylerYep/torchinfo) 라이브러리를 가져와 모델의 구조와 차원을 살펴보기 위해 사용# BioHealth_Data_AI_Competition
