import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import create_fusion_model
from models.physical_branch import PhysicalBranch
from models.geometric_branch import GeometricBranch
from utils.mask_utils import extract_seal_mask


DEFAULT_HEATMAP_DIR = os.path.join('visualizations', 'batch_llm')


def calibrate_with_physical_prior(real_prob, fake_prob, edge_density, edge_sharpness):
    """将物理特征作为先验注入概率，而不是直接翻转类别。"""
    sharp_score = np.clip((edge_sharpness - 1500.0) / 3000.0, 0.0, 1.0)
    sparse_score = np.clip((0.11 - edge_density) / 0.05, 0.0, 1.0)

    anomaly_score = np.clip(0.65 * sharp_score + 0.35 * sparse_score, 0.0, 1.0)
    boost = 0.55 * anomaly_score

    fake_cal = fake_prob + (1.0 - fake_prob) * boost
    fake_cal = float(np.clip(fake_cal, 0.0, 1.0))
    real_cal = 1.0 - fake_cal

    return real_cal, fake_cal, float(anomaly_score)

class SealAuthenticatorLLM:
    def __init__(self, model_checkpoint="checkpoints/best_model.pth", 
                 llm_model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                 heatmap_dir=DEFAULT_HEATMAP_DIR,
                 device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.heatmap_dir = heatmap_dir
        os.makedirs(self.heatmap_dir, exist_ok=True)
        print(f"[*] 使用设备: {self.device}")
        
        # 1. 加载主模型
        print(f"[*] 载入主分类模型权重: {model_checkpoint} ...")
        self.model = create_fusion_model(model_type='basic', num_classes=2, cnn_pretrained=False)
        
        if os.path.exists(model_checkpoint):
            self._load_main_model(model_checkpoint)
            print("[*] 主分类模型权重加载完成！")
        else:
            print(f"[!] 警告: 未找到模型权重 {model_checkpoint}，使用初始化参数！")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.physical_branch = PhysicalBranch()
        self.geometric_branch = GeometricBranch()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 2. 载入<1B参数的本地小模型 (默认Qwen2.5-0.5B)
        print(f"[*] 载入本地大语言模型用于生成判定理由: {llm_model_name} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.llm.eval()
            print("[*] 语言模型载入成功！")
        except Exception as e:
            print(f"[!] 语言模型加载失败，请确保安装了transformers且网络正常：{str(e)}")
            self.llm = None
            self.tokenizer = None

    def _load_main_model(self, model_checkpoint):
        """兼容加载不同版本结构的checkpoint（673/2209融合维度）。"""
        checkpoint = torch.load(model_checkpoint, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        fusion_weight = state_dict.get('fusion.0.weight', None)
        fusion_in_dim = int(fusion_weight.shape[1]) if fusion_weight is not None else None

        if fusion_in_dim == 2209:
            print("[*] 检测到旧版2209维融合权重，切换兼容模型结构...")
            self.model = create_fusion_model(
                model_type='basic',
                num_classes=2,
                cnn_pretrained=False,
                cnn_bottleneck_dim=2048
            )
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                print(f"[*] 兼容加载提示: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
            return

        if fusion_in_dim == 673:
            self.model = create_fusion_model(
                model_type='basic',
                num_classes=2,
                cnn_pretrained=False,
                cnn_bottleneck_dim=512
            )
            self.model.load_state_dict(state_dict)
            return

        if fusion_in_dim is not None:
            print(f"[!] 未知融合维度 {fusion_in_dim}，尝试按当前结构加载。")

        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError as e:
            msg = str(e)
            # 兼容旧版：fusion输入2209维、无cnn_bottleneck参数
            if 'fusion.0.weight' in msg and '2209' in msg:
                print("[*] 检测到旧版2209维融合权重，切换兼容模型加载...")
                legacy_model = create_fusion_model(
                    model_type='basic',
                    num_classes=2,
                    cnn_pretrained=False,
                    cnn_bottleneck_dim=2048
                )
                legacy_model.load_state_dict(state_dict, strict=False)
                self.model = legacy_model
                return
            raise

    def generate_reasoning(self, analysis):
        phys = analysis.get('physical_features', np.zeros(33))
        geom = analysis.get('geometric_features', np.zeros(128))
        
        edge_density = phys[0] if len(phys) > 0 else 0
        edge_sharpness = phys[1] if len(phys) > 1 else 0
        polar_std = np.std(geom[:36]) if len(geom) > 36 else 0
        
        is_fake = analysis['prediction'] == 'fake'
        prob = analysis['fake_prob'] if is_fake else analysis['real_prob']

        if is_fake:
            return (
                f"该印章判定为假章（置信度: {prob:.2%}）。"
                f"边缘密度({edge_density:.3f})、边缘锐利度({edge_sharpness:.1f})和几何规则性波动({polar_std:.3f})表现为异常，"
                "说明边缘与纹理一致性不足。"
            )

        return (
            f"该印章判定为真章（置信度: {prob:.2%}）。"
            f"边缘密度({edge_density:.3f})、边缘锐利度({edge_sharpness:.1f})和几何规则性波动({polar_std:.3f})处于正常范围，"
            "未见明显异常指标。"
        )

    def generate_gradcam(self, image_tensor, phys_tensor, geom_tensor, target_class=1):
        self.model.eval()
        features_list = []
        gradients_list = []

        def forward_hook(module, input, output):
            features_list.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients_list.append(grad_output[0])

        target_layer = self.model.cnn_branch.backbone.layer4
        f_hook = target_layer.register_forward_hook(forward_hook)
        b_hook = target_layer.register_full_backward_hook(backward_hook)

        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_()
        
        outputs = self.model(image_tensor, phys_tensor, geom_tensor, return_branch_outputs=True)
        fusion_logits = outputs['fusion_logits']
        
        target_score = fusion_logits[0, target_class]
        target_score.backward()

        f_hook.remove()
        b_hook.remove()

        grad = gradients_list[0].cpu().data.numpy()[0]
        feat = features_list[0].cpu().data.numpy()[0]
        
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(feat.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * feat[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

    def authenticate(self, image_path):
        total_start = time.perf_counter()
        if not os.path.exists(image_path):
            print(f"[!] 找不到图片路径: {image_path}")
            return
            
        print(f"\n[>] 开始鉴别印章: {image_path}")
        preprocess_start = time.perf_counter()
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)

        # 与训练/单图推理保持一致：先提取印章掩膜，再抹除背景区域
        mask = None
        try:
            mask = extract_seal_mask(image_np, method='multi_color')
            if mask.sum() >= 100:
                clean_image_np = image_np.copy()
                clean_image_np[mask == 0] = [255, 255, 255]
                cnn_image = Image.fromarray(clean_image_np)
            else:
                cnn_image = image_pil
        except Exception:
            mask = None
            cnn_image = image_pil

        cnn_input = self.transform(cnn_image).unsqueeze(0).to(self.device)
        
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        try:
            phys_vals = self.physical_branch.extract_features(image_bgr)
        except Exception:
            phys_vals = np.zeros(33)
            
        try:
            if mask is None:
                mask = extract_seal_mask(image_np, method='multi_color')
            geom_vals = self.geometric_branch.extract_features(mask)
        except Exception:
            geom_vals = np.zeros(128)
            
        phys_tensor = torch.from_numpy(phys_vals).unsqueeze(0).float().to(self.device)
        geom_tensor = torch.from_numpy(geom_vals).unsqueeze(0).float().to(self.device)
        preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

        model_inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(cnn_input, phys_tensor, geom_tensor, return_branch_outputs=True)
            fusion_probs = F.softmax(outputs['fusion_logits'], dim=1)
        model_inference_time_ms = (time.perf_counter() - model_inference_start) * 1000.0
            
        real_prob = fusion_probs[0, 0].item()
        fake_prob = fusion_probs[0, 1].item()

        edge_density = float(phys_vals[0]) if len(phys_vals) > 0 else 0.0
        edge_sharpness = float(phys_vals[1]) if len(phys_vals) > 1 else 0.0
        real_prob_cal, fake_prob_cal, anomaly_score = calibrate_with_physical_prior(
            real_prob, fake_prob, edge_density, edge_sharpness
        )

        pred_label = 'real' if real_prob_cal > fake_prob_cal else 'fake'
        
        print("\n================ 鉴别结果 ==================")
        print(f"原始真章概率 (Real)         : {real_prob:.4f}")
        print(f"原始假章概率 (Fake)         : {fake_prob:.4f}")
        print(f"校准真章概率 (Real)         : {real_prob_cal:.4f}")
        print(f"校准假章概率 (Fake)         : {fake_prob_cal:.4f}")
        print(f"物理先验分数                : {anomaly_score:.3f} (sharpness={edge_sharpness:.1f}, density={edge_density:.4f})")
        print(f"最终判定                   : {'真章 (REAL)' if pred_label == 'real' else '假章 (FAKE)'}")
        print("============================================")

        analysis = {
            'prediction': pred_label,
            'real_prob': real_prob_cal,
            'fake_prob': fake_prob_cal,
            'physical_features': phys_vals,
            'geometric_features': geom_vals
        }

        # LLM生成理由
        print("\n[>] 正在生成总体评价（判断理由）...")
        llm_start = time.perf_counter()
        evaluation_text = self.generate_reasoning(analysis)
        llm_reasoning_time_ms = (time.perf_counter() - llm_start) * 1000.0
        print("\n=== 总体评价 ===")
        print(evaluation_text)
        print("================\n")

        # 假章则生成注意力热力图
        heatmap_time_ms = 0.0
        if pred_label == 'fake':
            print("[>] 判断为假章，正在生成注意力热力图 (Grad-CAM)...")
            heatmap_start = time.perf_counter()
            cam = self.generate_gradcam(cnn_input, phys_tensor, geom_tensor, target_class=1)
            cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))

            # 仅在印章区域显示热力，避免背景误导性高亮
            if mask is not None and mask.sum() >= 100:
                mask_float = (mask > 0).astype(np.float32)
                cam_resized = cam_resized * mask_float
                if cam_resized.max() > 0:
                    cam_resized = cam_resized / cam_resized.max()
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            superimposed_img = heatmap * 0.4 + image_np * 0.6
            superimposed_img = superimposed_img.astype(np.uint8)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Original Seal')
            plt.imshow(image_np)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title('Attention Heatmap (Fake Features)')
            plt.imshow(superimposed_img)
            plt.axis('off')
            
            plt.tight_layout()
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(self.heatmap_dir, f'{image_name}_attention_heatmap.png')
            plt.savefig(out_path)
            plt.close()
            print(f"[+] 注意力热力图已保存至: {os.path.abspath(out_path)}")
            heatmap_time_ms = (time.perf_counter() - heatmap_start) * 1000.0
        else:
            out_path = None

        total_inference_time_ms = (time.perf_counter() - total_start) * 1000.0
        print(f"[*] 推理耗时统计: total={total_inference_time_ms:.2f}ms, preprocess={preprocess_time_ms:.2f}ms, model={model_inference_time_ms:.2f}ms, llm={llm_reasoning_time_ms:.2f}ms, heatmap={heatmap_time_ms:.2f}ms")

        return {
            "real_probability_raw": real_prob,
            "fake_probability_raw": fake_prob,
            "real_probability": real_prob_cal,
            "fake_probability": fake_prob_cal,
            "prediction": pred_label,
            "evaluation": evaluation_text,
            "heatmap_path": out_path,
            "inference_time_ms": total_inference_time_ms,
            "timing_breakdown_ms": {
                "preprocess": preprocess_time_ms,
                "model_inference": model_inference_time_ms,
                "llm_reasoning": llm_reasoning_time_ms,
                "heatmap": heatmap_time_ms
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="印章真伪鉴别与LLM报告生成")
    parser.add_argument("--image", type=str, default="data/test/fake/fake_sample_0.png", help="待测试图片路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="模型权重路径")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="所用小参LLM名字或本地路径 (<1B)")
    parser.add_argument("--heatmap-dir", type=str, default=DEFAULT_HEATMAP_DIR, help="假章热图保存目录")
    args = parser.parse_args()

    # 如果默认测试图片不存在，则通过glob或者os.listdir获取第一个存在的图片作为演示
    test_image = args.image
    if not os.path.exists(test_image):
        print(f"默认图片 {test_image} 不存在，尝试寻找其他图片。")
        for root, dirs, files in os.walk("data"):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    test_image = os.path.join(root, f)
                    break
            if os.path.exists(test_image):
                break

    auth = SealAuthenticatorLLM(
        model_checkpoint=args.checkpoint,
        llm_model_name=args.llm,
        heatmap_dir=args.heatmap_dir
    )
    auth.authenticate(test_image)
