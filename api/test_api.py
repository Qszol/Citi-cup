import os
from seal_api import SealAPI

def main():
    # 模拟在Web应用启动时的操作：初始化封好的对象
    print("[*] 正在加载印章模型，请稍候...")
    model_path = "../checkpoints/best_model.pth"
    
    # 填入测试图（如需测试真实的请改成本地真实的图片路径）
    test_img = "../data/test/fake/fake_sample_0.png"
    
    # 如果找不到模型和图片，尝试寻找其他可用文件进行测试
    if not os.path.exists(model_path):
        print(f"[!] 找不到权重文件 {os.path.abspath(model_path)}")
        return
        
    seal_evaluator = SealAPI(checkpoint_path=model_path)
    
    if not os.path.exists(test_img):
        # 兜底找一张图测试
        for root, dirs, files in os.walk("../data"):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    test_img = os.path.join(root, f)
                    break
            if os.path.exists(test_img):
                break
    
    print(f"\n[*] 正在鉴定印章: {test_img}")
    # 模拟Web收到请求进行鉴别
    try:
        response_json = seal_evaluator.verify_seal(test_img)
        print("\n\n======== 接口返回 JSON ========")
        for key, value in response_json.items():
            print(f"{key}: {value}")
        print("===============================\n")
    except Exception as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    main()
