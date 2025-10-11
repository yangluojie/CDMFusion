import torch
import utils
import os
import math
import time
import torchvision.transforms as transforms
from utils import metrics
from typing import Dict, Any, List, Tuple, NamedTuple
import utils.logging

def data_transform(X):
    return 2 * X - 1.0

def sigmoid_zheng(x): 
    return  ((1 / (1 + math.exp(-(x))))-0.5)*2

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class Diffusion_Generation:
    def __init__(self, diffusion, args, config):
        super(Diffusion_Generation, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def generation(self, val_loader):
        image_folder = os.path.join(self.args.image_folder)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print(f"starting processing from image {y[0]}")                
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                pipeline = ParameterSelectionOrchestrator()
                canshu = pipeline.process(x)
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_cond2 = x[:, 6:, :, :].to(self.diffusion.device)
                x_cond = data_transform(x_cond)
                x_cond2 = data_transform(x_cond2)
                x_T = torch.randn(x_cond.size(), device=self.diffusion.device)
                if self.config.data.dataset=='RSFB': stage = 8
                else: stage = 4
                x_output = self.diffusion.sample_image(x_cond,x_cond2, x_T, stage, canshu)
                x_outputsave = inverse_data_transform(x_output)
                utils.logging.save_image(x_outputsave, os.path.join(image_folder, f"{y[0]}.png"))

class SpectralSpatialFeatureExtractor: 
    def __init__(self):
        self._to_pil_image = transforms.ToPILImage()

    def _convert_tensor_to_modalities(self, tensor_batch) -> Tuple[Any, Any]:
        vi_img = self._to_pil_image(tensor_batch[0, :3, :, :])
        ir_img = self._to_pil_image(tensor_batch[0, 6:, :, :])
        return vi_img, ir_img

    def analyze(self, tensor_batch) -> Dict[str, float]:
        vi_img, ir_img = self._convert_tensor_to_modalities(tensor_batch)
        return {
            "vi_SD": metrics.SD_function(vi_img),
            "ir_SD": metrics.SD_function(ir_img),
            "vi_CC": metrics.CC_function(vi_img, ir_img, vi_img),
            "jun_vi": metrics.junzhi(vi_img),
            "jun_ir": metrics.junzhi(ir_img),
            "_t": time.time()
        }

class ParameterVector(NamedTuple):
    p0: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float

    def to_list(self) -> List[float]:
        return [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]

class ParameterDomainRepository:
    def __init__(self):
        self._domain_map: Dict[str, ParameterVector] = {
            "base": ParameterVector(0, 1.3, 1.4, 0.9, 0, 0),
            "cc_mod": ParameterVector(0, 1.3, 1.4, 0.9, 0.1, 0),
            "hi_ir_vi": ParameterVector(0, 1.6, 1.4, 0.3, -0.1, 1),
            "low_ir": ParameterVector(0.6, 0, 1, 1.56, 0.63, 0)
        }
        self._aliases = {
            "branch_cc": "cc_mod",
            "branch_high_ir_vi": "hi_ir_vi",
            "branch_dark_ir": "low_ir",
            "branch_default": "base"
        }

    def fetch(self, branch_name: str) -> ParameterVector:
        key = self._aliases.get(branch_name, branch_name)
        return self._domain_map.get(key, self._domain_map["base"])

class AdaptiveConditionalGatingUnit:
    def __init__(self, param_repo: ParameterDomainRepository):
        self._repo = param_repo
        self._log: List[Tuple[str, float]] = []

    def _stage1_cc_gate(self, f: Dict[str, float], vec: ParameterVector) -> ParameterVector:
        gate_signal = (f["vi_CC"] - 0.45) > 0
        p4_new = vec.p4 + gate_signal * (0.1 - vec.p4)
        self._record_state("STAGE1_CC", gate_signal)
        return ParameterVector(vec.p0, vec.p1, vec.p2, vec.p3, p4_new, vec.p5)

    def _stage2_high_ir_vi(self, f: Dict[str, float], vec: ParameterVector) -> ParameterVector:
        cond = (f["jun_ir"] - 105 > 0) * (f["jun_vi"] - 90 > 0) * (f["ir_SD"] - f["vi_SD"] > 0)
        selected_vec = self._repo.fetch("hi_ir_vi") if cond else vec
        self._record_state("STAGE2_HI_IR_VI", cond)
        return selected_vec

    def _stage3_low_ir(self, f: Dict[str, float], vec: ParameterVector) -> ParameterVector:
        cond = (f["jun_ir"] - 50) < 0
        selected_vec = self._repo.fetch("low_ir") if cond else vec
        self._record_state("STAGE3_LOW_IR", cond)
        return selected_vec

    def decide_branch(self, features: Dict[str, Any]) -> List[float]:
        v0 = self._repo.fetch("base")
        v1 = self._stage1_cc_gate(features, v0)
        v2 = self._stage2_high_ir_vi(features, v1)
        v3 = self._stage3_low_ir(features, v2)
        return v3.to_list()

    def _record_state(self, stage_record: str, condition_result: Any):
        self._log.append((stage_record, bool(condition_result)))


class ParameterSelectionOrchestrator:
    def __init__(self):
        self._feature_extractor = SpectralSpatialFeatureExtractor()
        self._param_repo = ParameterDomainRepository()
        self._gating_unit = AdaptiveConditionalGatingUnit(self._param_repo)

    def process(self, x_tensor) -> List[float]:
        features = self._feature_extractor.analyze(x_tensor)
        return self._gating_unit.decide_branch(features)


class NoiseScheduler:
    def __init__(self, beta_schedule):
        self.beta = torch.cat([torch.zeros(1).to(beta_schedule.device), beta_schedule], dim=0)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)

class TrajectoryPlanner:
    def __init__(self, total_time_steps=1000, tau1_ratio=0.7, tau2_ratio=0.2):
        self.T = total_time_steps
        self.tau1_ratio = tau1_ratio
        self.tau2_ratio = tau2_ratio

    def nonlinear_time_mapping(self, ratio):
        return int(self.T * ratio)

    def get_schedule(self):
        tau1 = self.nonlinear_time_mapping(self.tau1_ratio)
        tau2 = self.nonlinear_time_mapping(self.tau2_ratio)
        return [tau1, tau2], [-1, -1]

class ConditionalDenoiser:
    def __init__(self, model):
        self.model = model

    def forward(self, xt, t, x_cond, x_cond2, w, stage, canshu):
        if x_cond2.size()[1] == 1:
            x_cond2 = x_cond2.repeat(1, 3, 1, 1)

        if w == 0:
            return self.model(torch.cat([x_cond, x_cond2, xt], dim=1), t, stage, canshu)

        et_c = self.model(torch.cat([x_cond, x_cond2, xt], dim=1), t, stage, canshu)
        ir0 = torch.zeros_like(x_cond2) - 1.0
        et_un = self.model(torch.cat([x_cond, ir0, xt], dim=1), t, stage, canshu)
        return (1 + w) * et_c - w * et_un

class LatentReconstructor:
    def reconstruct(self, xt, et, at):
        return (xt - et * (1 - at).sqrt()) / at.sqrt()

class LeapAndPatrolExecutor:
    def __init__(self, planner, scheduler, denoiser, reconstructor):
        self.planner = planner
        self.scheduler = scheduler
        self.denoiser = denoiser
        self.reconstructor = reconstructor

    def execute(self, x, x_cond, x_cond2, canshu, stage, eta=0.):
        with torch.no_grad():
            n = x.size(0)
            xs = [x]
            x0_preds = []

            t_list, tnext_list = self.planner.get_schedule()
            torch_0 = torch.ones(n).to(x.device)
            w = canshu[0]

            for idx in range(len(t_list)):
                t = torch_0 + t_list[idx]
                next_t = torch_0 + tnext_list[idx]
                at = self.scheduler.alpha(t.int())
                at_next = self.scheduler.alpha(next_t.long())
                xt = xs[-1].to(x.device)
                et = self.denoiser.forward(xt, t, x_cond, x_cond2, w, stage, canshu)
                x0_t = self.reconstructor.reconstruct(xt, et, at)
                c2 = ((1 - at_next)).sqrt()
                xt_next = at_next.sqrt() * x0_t + c2 * et

                xs.append(xt_next.to(x.device))
            return xs, x0_preds

def generalized_steps(x, x_cond, x_cond2, canshu,stage, model, b, eta=0.):
    scheduler = NoiseScheduler(b)
    planner = TrajectoryPlanner(total_time_steps=1000, tau1_ratio=0.7, tau2_ratio=0.2)
    denoiser = ConditionalDenoiser(model)
    reconstructor = LatentReconstructor()
    executor = LeapAndPatrolExecutor(planner, scheduler, denoiser, reconstructor)

    return executor.execute(x, x_cond, x_cond2, canshu,stage, eta)
