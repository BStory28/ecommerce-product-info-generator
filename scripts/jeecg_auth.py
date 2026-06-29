#!/usr/bin/env python3
"""
JeecgBoot OpenAPI 签名认证工具

用法:
  from jeecg_auth import JeecgAuth

  auth = JeecgAuth(app_key="xxx", app_secret="xxx")
  headers = auth.get_headers()
  # headers = {"X-Tenant-Id": "1000", "appkey": "...", "signature": "...", "timestamp": "..."}

环境变量:
  AIGC_API_BASE       = https://aigc.hkttok.com (默认)
  AIGC_APP_KEY        = 你的 appKey
  AIGC_APP_SECRET     = 你的 appSecret
  AIGC_TENANT_ID      = 1000 (默认)

签名算法:
  sign_str = appKey + appSecret + timestamp(13位毫秒)
  signature = MD5(sign_str).hexdigest()  # 小写32位
"""

import hashlib
import os
import time
from pathlib import Path


def _load_env():
    """Auto-load .env from project root or parent dirs"""
    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir] + list(script_dir.parents):
        env_path = parent / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    # Strip inline comments (everything after #)
                    val = val.strip().strip("\"'")
                    comment_pos = val.find(" #")
                    if comment_pos > 0:
                        val = val[:comment_pos].rstrip()
                    elif val.startswith("#"):
                        continue
                    if key and not os.environ.get(key):
                        os.environ[key] = val
            break


_load_env()


class JeecgAuth:
    """JeecgBoot OpenAPI 签名认证"""

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        api_base: str = "",
        tenant_id: str = "",
    ):
        self.app_key = app_key or os.environ.get("AIGC_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("AIGC_APP_SECRET", "")
        self.api_base = (api_base or os.environ.get("AIGC_API_BASE", "https://aigc.hkttok.com/api")).rstrip("/")
        self.tenant_id = tenant_id or os.environ.get("AIGC_TENANT_ID", "1000")

    def validate(self):
        """Raise ValueError if credentials are missing"""
        if not self.app_key:
            raise ValueError("AIGC_APP_KEY 未设置。请在 .env 中设置: AIGC_APP_KEY=your_key")
        if not self.app_secret:
            raise ValueError("AIGC_APP_SECRET 未设置。请在 .env 中设置: AIGC_APP_SECRET=your_secret")

    def sign(self) -> tuple[str, str]:
        """计算签名，返回 (timestamp, signature)"""
        self.validate()
        timestamp = str(int(time.time() * 1000))
        sign_str = self.app_key + self.app_secret + timestamp
        signature = hashlib.md5(sign_str.encode("utf-8")).hexdigest()
        return timestamp, signature

    def get_headers(self) -> dict:
        """获取带签名的请求头"""
        timestamp, signature = self.sign()
        return {
            "X-Tenant-Id": self.tenant_id,
            "appkey": self.app_key,
            "signature": signature,
            "timestamp": timestamp,
            "Content-Type": "application/json",
        }

    def url(self, path: str) -> str:
        """拼接完整 URL"""
        return f"{self.api_base}/jeecg-boot/openapi/call/{path.lstrip('/')}"

    def image_submit_url(self) -> str:
        return self.url("generation/image")

    def image_query_url(self) -> str:
        return self.url("generation/image")

    def video_submit_url(self) -> str:
        return self.url("generation/video")

    def video_query_url(self) -> str:
        return self.url("generation/video")

    def models_url(self) -> str:
        return self.url("models")
