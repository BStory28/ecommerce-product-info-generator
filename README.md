# 产品卖点基础信息生成器

> **Version:** 1.2.0 | [SKILL.md](SKILL.md) | [Changelog](#)

国际化电商视频管线 **Skill1** — 根据产品图片和商品信息自动识别产品类别、凝练结构化卖点、推断适用人群和场景。

## 管线定位

```
Skill1 ← 本技能       → 输出: product_layer.png + selling_points.json
         ↓
Skill2: ecommerce-video-script-generator → 分镜脚本
         ↓
Skill3: ecommerce-video-generator → 最终视频
```

## 功能

- 基于产品白底图自动识别产品类别和核心品类
- 凝练结构化卖点（功能卖点 + 情感卖点 + 差异化卖点）
- 推断适用人群和使用场景
- 根据目标市场调整语言和文化适配
- 生成产品白底图（`product_layer.png`）和结构化卖点数据（`selling_points.json`）

## 快速开始

```bash
# 纯 Python 标准库，无第三方依赖
python scripts/generate_base_image.py \
  --product 产品图片.png \
  --name "产品名称" \
  --country "目标国家" \
  --video-type "视频类型" \
  --duration 30 \
  --output ./output
```

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--product` | ✅ | 产品白底图路径 |
| `--name` | ✅ | 产品名称（目标市场语言） |
| `--country` | ✅ | 目标国家/市场 |
| `--video-type` | ✅ | 视频类型 |
| `--duration` | ✅ | 视频时长（秒） |
| `--hook-points` | ❌ | 吸睛点描述 |
| `--price` | ❌ | 福利价格 + 货币单位 |
| `--output` | ❌ | 输出目录（默认 ./output） |

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/product_layer.png` | 产品白底图（下载产品图片并重命名） |
| `output/selling_points.json` | 结构化卖点数据（供 Skill2 消费） |

## 上下游

- **下游**: [ecommerce-video-script-generator](https://github.com/BStory28/ecommerce-video-script-generator) — 消费本技能输出的卖点数据生成分镜脚本
- **SDK**: 暂无需额外 SDK，依赖 Python 标准库 + `requests`

## 注意事项

- `product_layer.png` 为直接引用的产品图片（不进行AI生成修改）
- `selling_points.json` 包含 `product_function` + `user_pain_point` 字段供 Skill2 使用
- 执行完成后 AI 会自动询问是否继续执行 Skill2
