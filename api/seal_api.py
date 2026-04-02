import sys
import os

# 将项目根目录添加到系统路径，以便能够导入根目录下的 visualize_prediction_llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize_prediction_llm import SealAuthenticatorLLM

class SealAPI:
    """
    提供给Web后端调用的统一接口封装类。
    包含了模型的初始化、图像验证和结果返回的功能。
    """
    def __init__(self, checkpoint_path, llm_model_name="Qwen/Qwen2.5-0.5B-Instruct", heatmap_dir="visualizations/batch_llm"):
        """
        初始化印章鉴伪API。
        该过程会加载权重文件和LLM模型，如果后端是在启动时初始化，建议将其作为全局变量。
        
        Args:
            checkpoint_path (str): 模型权重路径 (.pth)，例如 ../checkpoints/best_model.pth
            llm_model_name (str): 提供大语言模型在本地的路径或 HuggingFace的Repo ID
            heatmap_dir (str): 假章热力图输出目录（仅 prediction=fake 时生成）
        """
        self.authenticator = SealAuthenticatorLLM(
            model_checkpoint=checkpoint_path, 
            llm_model_name=llm_model_name,
            heatmap_dir=heatmap_dir
        )

    def verify_seal(self, image_path):
        """
        传入印章图像，判断真伪。
        
        Args:
            image_path (str): 需要验证的图像文件在服务器本地的绝对或相对路径
            
        Returns:
            dict: 包含以下字段：
                  - real_probability (float): 该印章判断为真实印章的概率(0~1)
                  - fake_probability (float): 该印章判断为伪造印章的概率(0~1)
                  - prediction (str): "real" 或 "fake"
                  - evaluation (str): 大语言模型生成的分析评价（一段话）
                  - heatmap_path (str): 注意力热力图图像的保存路径（仅判定为假章时返回图像，否则为None）
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        result = self.authenticator.authenticate(image_path)
        return result
