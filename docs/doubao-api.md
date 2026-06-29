# 基图生成 API 接口文档

## 概述

本管线通过 **aigc.hkttok.com JeecgBoot OpenAPI** 调用图片生成模型（如 GPT Image、Seedream 等）
进行电商基图生成。支持图生图（产品保真）和文生图（人物生成）两种模式。

---

## 1. 认证方式

详见 `scripts/jeecg_auth.py`。

所有请求通过 `/jeecg-boot/openapi/call/{path}` 网关转发，携带签名头：

```
X-Tenant-Id: 1000
appkey: {app_key}
signature: MD5(app_key + app_secret + timestamp).hexdigest()
timestamp: {13位毫秒时间戳}
Content-Type: application/json
```

---

## 2. 提交图片生成任务

### Endpoint

```
POST /jeecg-boot/openapi/call/generation/image/submit
```

### Request

```json
{
  "model": "{model_id（从 models 接口获取）}",
  "prompt": "产品展示图，白色磨砂瓶身，金色字体标签，纯净背景",
  "negative_prompt": "文字，水印，杂乱背景，其他产品",
  "image": "{base64_encoded}",
  "image_resolution": "1080x1920",
  "n": 1,
  "size": "1080x1920",
  "response_format": "b64_json",
  "parameters": {
    "product_preservation_level": "high"
  }
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型 ID（通过 /models 端点查询） |
| prompt | string | 是 | 描述文本 |
| negative_prompt | string | 否 | 负面提示词 |
| image | string | 否（图生图必填） | 参考图片 base64 或 URL |
| image_resolution | string | 否 | 参考图分辨率 |
| n | int | 否 | 生成数量（默认1） |
| size | string | 否 | 输出尺寸（默认 1080x1920） |
| response_format | string | 否 | b64_json/url |
| parameters.product_preservation_level | string | 否 | 产品保真度（high/low） |

### Response

```json
{
  "id": "task_xxxxxxxxxx",
  "status": "pending"
}
```

返回 `id` 用于查询结果。

---

## 3. 查询生成结果

### Endpoint

```
GET /jeecg-boot/openapi/call/generation/image/query?id={taskId}
```

### Response

```json
{
  "status": "SUCCESS",
  "url": "https://cdn.hkttok.com/images/xxx.png",
  ...
}
```

| 状态 | 含义 |
|------|------|
| SUCCESS | 生成完成 |
| FAILED | 生成失败 |
| RUNNING | 生成中 |
| PENDING | 队列等待 |

---

## 4. 查询可用模型

### Endpoint

```
GET /jeecg-boot/openapi/call/models
```

查询结果中的图片生成模型可用于本模块。

```bash
python scripts/query_models.py
```

---

## 5. 生成策略

### 层① 产品白底图（图生图）

- 使用 `product_preservation_level=high` 保真产品外观
- 传入产品白底图作为 `image` 参考
- Prompt 描述目标市场风格调整

### 层③ 人物图（文生图）

- 不传入 `image`，纯文本生成
- Prompt 描述人物特征、动作、表情
- 生成透明背景 PNG

---

## 6. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AIGC_API_BASE` | API 基础地址 | `https://aigc.hkttok.com` |
| `AIGC_APP_KEY` | 应用 Key | — |
| `AIGC_APP_SECRET` | 应用 Secret | — |
| `AIGC_TENANT_ID` | 租户 ID | `1000` |
| `AIGC_IMAGE_MODEL` | 图片模型 ID | `2049087333668446209`（Seedream 5.0 Lite） |
