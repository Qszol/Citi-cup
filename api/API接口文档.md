# 印章鉴伪模型开发集成文档（Web开发者指南）

这份文档专为 Web / 软件后端开发者编写，供您快速在您的应用程序中集成和调用我们的印章鉴伪模型，并且包含所有的必要的代码示例和接口描述。

## 目录
1. [模型架构与工作流](#1-模型架构与工作流)
2. [环境依赖](#2-环境依赖)
3. [核心接口 (Python)](#3-核心接口-python)
4. [FastAPI 后端参考示例](#4-fastapi-后端参考示例)
5. [常见问题](#5-常见问题)

---

## 1. 模型架构与工作流

当用户在前端上传一张印章图片时，系统会：
1. **多模态印章真伪分析**：通过包含卷积特征、物理边缘特征、极坐标几何特征的三分支融合模型来分析出印章的真实度（概率）。
2. **大语言模型（LLM）生成报告**：将上述判断的输出传给本地部署的大模型，它将输出**一段自然语言的总体评价**作为判别理由。
3. **缺陷可视化（Grad-CAM）**：如果印章被判定为假（伪造），系统会自动生成其对应的注意力热力图片（标记假章的造假可能位置）。

您只需要使用本文件夹下提供的类。这些底层复杂逻辑都已经被我们封装好！

## 2. 环境依赖
在运行模型之前，请确保您的 Python 环境安装了必要的包：
```bash
# 请回到项目根目录
pip install -r requirements.txt
```

### 2.1 千问模型前置要求（必须）
- 默认会加载 `Qwen/Qwen2.5-0.5B-Instruct` 用于生成 `evaluation` 文本。
- 首次运行若使用 Hugging Face Repo ID，需要可访问外网下载模型权重。
- 离线环境必须提前下载模型到本地目录，并在初始化时通过 `llm_model_name` 传入本地路径。
- 如果无法加载 LLM，主模型真假分类仍可执行，但 `evaluation` 会退化为固定提示（例如“语言模型未正确加载。”）。

上线前检查清单（建议写入部署流程）：
- 检查一：`llm_model_name` 是否可访问（Repo ID 或本地目录路径有效）。
- 检查二：首次启动机器是否具备下载权限；若无权限，是否已完成离线模型预置。
- 检查三：服务启动日志中是否出现“语言模型载入成功”。

---

## 3. 核心接口 (Python)

我们为您提供了一个极其简洁易用的包装类：`api/seal_api.py` 中的 `SealAPI` 类。

### 3.1 实例化模型
后端启动时（例如您的 Flask、Django、FastAPI 或者在主进程初始化处），您需要进行模型预加载。

```python
from api.seal_api import SealAPI

# 传入PyTorch模型权重，以及采用的大语言模型路径
seal_model = SealAPI(
    checkpoint_path="../checkpoints/best_model.pth", 
  llm_model_name="Qwen/Qwen2.5-0.5B-Instruct",  # 可选填 HuggingFace Repo 或本地下载好的路径
  heatmap_dir="../visualizations/batch_llm"      # 可选，假章热图输出目录
)
```
离线部署示例：
```python
seal_model = SealAPI(
    checkpoint_path="../checkpoints/best_model.pth",
    llm_model_name=r"D:/models/Qwen2.5-0.5B-Instruct",  # 本地模型目录
    heatmap_dir="../visualizations/batch_llm"
)
```
*> **建议**：由于模型比较庞大，载入需时间，请务必保证在系统进程初始化时只加载这1次，避免每次请求网络重复加载。*

### 3.2 验证图片并获取返回值
您只需要调用 `verify_seal(图片路径)` 方法：

```python
result = seal_model.verify_seal("你的图片路径.png")
print(result)
```

**期望的返回值 `dict` JSON 格式：**
```json
{
  "real_probability": 0.0521,
  "fake_probability": 0.9479,
  "prediction": "fake",
  "evaluation": "系统检测该印章为假章（置信度 94.79%）。物理边缘特征指标（密度: 4.870，锐利度:105.7，规则性:0.297）。该印章在边缘和纹理规则上展现出了明显伪造破绽和模型异常点，存在极高人工模仿痕迹并缺少正常手盖章带来的墨迹自然渗透现象。",
  "heatmap_path": "visualizations/batch_llm/sample_attention_heatmap.png",
  "inference_time_ms": 318.42,
  "timing_breakdown_ms": {
    "preprocess": 41.25,
    "model_inference": 22.87,
    "llm_reasoning": 0.18,
    "heatmap": 254.12
  }
}
```

- **`real_probability`** (Float)：真章判定概率 (0~1)
- **`fake_probability`** (Float)：假章判定概率 (0~1)
- **`prediction`** (String)：结论枚举，`"real"` 或 `"fake"`
- **`evaluation`** (String)：LLM 专门生成的一段通俗易懂的报告（用作反馈给前端前端展示给用户看）。
- **`heatmap_path`** (String 或 None)：只有判定为 `fake` 假时才会返回图像路径。默认输出目录是 `visualizations/batch_llm`，文件名为 `原图名_attention_heatmap.png`；判定为 `real` 时为 `None`。
- **`inference_time_ms`** (Float)：单次推理总耗时，单位毫秒。
- **`timing_breakdown_ms`** (Object)：分阶段耗时（毫秒），包含 `preprocess`、`model_inference`、`llm_reasoning`、`heatmap`。

---

## 4. FastAPI 后端参考示例
为了帮助您快速对接，我们在本文件夹中提供了 `fastapi_example.py` 示例后端。

**启动示例:**
```bash
cd api
python fastapi_example.py
```
这将在 `http://0.0.0.0:8000` 启动接口服务。你可以使用 Postman 或者前端代码访问：`POST /api/v1/verify_seal` 并通过 `form-data` 上传文件键为 `file` 的图片。它将接收您的图片、保存到临时目录然后调用算法并以 REST API 的标准 JSON 直接打给前端。

## 5. 常见问题
- **如果我得到 ModuleNotFoundError?** 
请检查您的执行目录。`seal_api.py` 会去包含根目录以读取 `visualize_prediction_llm`，如果您移动了项目架构，请同步修改它 `sys.path.append` 的代码即可。
- **关于 heatmap（热力图）保存问题?**
目前默认保存在项目下 `visualizations/batch_llm`。您也可以在初始化 `SealAPI` 时通过 `heatmap_dir` 参数指定目录。
