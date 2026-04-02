from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
import os
import uuid

# 引入我们封装好的SealAPI
from seal_api import SealAPI

app = FastAPI(title="印章鉴伪 API", description="印章鉴伪项目配套给Web端的接口", version="1.0")

# ==================== 初始化模型（全局）====================
# 注意：路径请根据您服务器部署时候的绝对路径进行修改
CHECKPOINT_PATH = "../checkpoints/best_model.pth"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
HEATMAP_DIR = "../visualizations/batch_llm"

print("[*] 正在加载印章鉴伪核心与大语言模型，请稍候...")
# 在FastAPI启动时预加载，保证API响应速度
if os.path.exists(CHECKPOINT_PATH):
    seal_model = SealAPI(
        checkpoint_path=CHECKPOINT_PATH,
        llm_model_name=LLM_MODEL_NAME,
        heatmap_dir=HEATMAP_DIR
    )
    print("[+] 模型加载完成！")
else:
    print(f"[-] 找不到模型权重 {CHECKPOINT_PATH}，如果在测试，请修改路径。")
    seal_model = None
# ==========================================================

class SealVerifyResponse(BaseModel):
    real_probability: float
    fake_probability: float
    prediction: str
    evaluation: str
    heatmap_path: str | None
    inference_time_ms: float
    timing_breakdown_ms: dict[str, float]

@app.post("/api/v1/verify_seal", response_model=SealVerifyResponse)
async def verify_seal_endpoint(file: UploadFile = File(...)):
    """
    接收用户上传的待鉴定印章图像，调用鉴别算法，返回JSON鉴定结果
    """
    if seal_model is None:
        raise HTTPException(status_code=500, detail="模型未成功加载，请检查后端模型配置路径。")
        
    # 保存上传的图片到服务器本地临时目录进行判断
    os.makedirs("temp_uploads", exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_filepath = os.path.join("temp_uploads", temp_filename)
    
    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # =========== 核心调用 ===========
        result = seal_model.verify_seal(temp_filepath)
        # ================================
        
        # 鉴定结束后，可以考虑清理缓存图片（为了演示此处暂时只留着做日记记录）
        # os.remove(temp_filepath)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"鉴别时出现异常: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
