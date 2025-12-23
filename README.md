DeZero-MLX: Deep Learning from Scratch 3 on Apple Silicon

DeZero-MLX는 사이토 고키의 저서 [『밑바닥부터 시작하는 딥러닝 3』](https://github.com/koki0702/dezero)에서 구현한 프레임워크인 **DeZero**를 Apple Silicon(M1/M2/M3/M4) 환경에서 GPU 가속이 가능하도록 개조한 프로젝트입니다.

기존 DeZero는 NVIDIA GPU(CUDA/CuPy)만을 지원하여 Mac 사용자들은 GPU 가속을 사용할 수 없었습니다. 
이 프로젝트는 Apple의 [MLX](https://github.com/ml-explore/mlx) 라이브러리를 백엔드로 도입하여, 
Mac에서도 DeZero의 모든 기능을 고속으로 학습할 수 있도록 포팅하였습니다.
