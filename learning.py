import math
import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # EMA 모델을 생성하고 초기화
        self.ema = deepcopy(model).eval()  # EMA 모델을 원본 모델로 초기화
        self.updates = updates  # 업데이트 횟수
        # 감쇠(decay) 값을 동적으로 계산하는 람다 함수
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.)) 
        for p in self.ema.parameters():
            p.requires_grad_(False)  # EMA 모델의 파라미터는 학습되지 않도록 설정

    def update(self, model):
        # EMA 모델의 파라미터를 업데이트
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)  # 동적으로 감쇠 값 계산

            msd = model.state_dict()  # 원본 모델의 상태 딕셔너리
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d  # EMA 모델의 기존 값에 감쇠 적용
                    v += (1. - d) * msd[k].detach()  # 원본 모델의 새로운 값을 EMA 모델에 반영
