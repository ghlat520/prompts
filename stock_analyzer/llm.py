"""LLM客户端 - 多提供商路由，支持T1/T2/T3三梯度"""

import os
import json
import time
import requests
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_yuan: float
    elapsed_ms: int


class LLMClient:
    """多提供商LLM客户端，支持自动降级"""

    # 提供商配置: (env_key, base_url, default_model, 价格/百万token)
    PROVIDERS = {
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com",
            "models": {
                "T1": "claude-haiku-4-5-20251001",
                "T2": "claude-sonnet-4-5-20250929",
                "T3": "claude-sonnet-4-5-20250929",
            },
            "price_input": 21.6,   # 元/百万token (Sonnet $3/M → ~21.6元)
            "price_output": 108.0,  # 元/百万token (Sonnet $15/M → ~108元)
        },
        "deepseek": {
            "env_key": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "models": {
                "T2": "deepseek-chat",       # DeepSeek-V3
                "T3": "deepseek-reasoner",   # DeepSeek-R1
            },
            "price_input": 2.0,   # 元/百万token
            "price_output": 8.0,
        },
        "baidu": {
            "env_key": "BAIDU_API_KEY",
            "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
            "models": {
                "T1": "ernie-speed-8k",
            },
            "price_input": 0,
            "price_output": 0,
        },
        "zhipu": {
            "env_key": "ZHIPU_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "models": {
                "T1": "glm-4-flash",
            },
            "price_input": 0,
            "price_output": 0,
        },
        "alibaba": {
            "env_key": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": {
                "T2": "qwen-plus",
                "T3": "qwen-max",
            },
            "price_input": 2.4,
            "price_output": 9.6,
        },
        "openai_compatible": {
            "env_key": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "model_env": "OPENAI_MODEL",  # 从环境变量读模型名
            "models": {
                "T1": "auto",
                "T2": "auto",
                "T3": "auto",
            },
            "price_input": 0,
            "price_output": 0,
        },
    }

    # 梯度 → 提供商优先级
    TIER_PRIORITY = {
        "T1": ["baidu", "zhipu", "anthropic", "openai_compatible"],
        "T2": ["deepseek", "alibaba", "anthropic", "openai_compatible"],
        "T3": ["deepseek", "alibaba", "anthropic", "openai_compatible"],
    }

    def __init__(self, max_tier: str = "T3"):
        self.max_tier = max_tier
        self._available = self._detect_available()

    def _detect_available(self) -> dict:
        """检测可用的提供商"""
        available = {}
        for name, cfg in self.PROVIDERS.items():
            key = os.environ.get(cfg.get("env_key", ""), "")
            if key:
                available[name] = key
        return available

    def available_providers(self) -> list[str]:
        return list(self._available.keys())

    def call(self, system_prompt: str, user_prompt: str, tier: str = "T2",
             temperature: float = 0.3, max_tokens: int = 8000) -> LLMResponse:
        """调用LLM，自动路由到合适的提供商"""
        # 降级: min(推荐tier, max_tier)
        effective_tier = self._min_tier(tier, self.max_tier)

        # 按优先级尝试提供商
        providers = self.TIER_PRIORITY.get(effective_tier, ["openai_compatible"])
        last_error = None

        for provider_name in providers:
            if provider_name not in self._available:
                continue

            try:
                return self._call_provider(
                    provider_name, effective_tier,
                    system_prompt, user_prompt,
                    temperature, max_tokens
                )
            except Exception as e:
                last_error = e
                print(f"  [WARN] {provider_name} 调用失败: {e}，尝试下一个...")
                continue

        raise RuntimeError(
            f"所有{effective_tier}提供商均不可用。"
            f"已检测API Key: {self.available_providers()}。"
            f"错误: {last_error}"
        )

    def _call_provider(self, provider_name: str, tier: str,
                       system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> LLMResponse:
        """调用具体提供商"""
        cfg = self.PROVIDERS[provider_name]
        api_key = self._available[provider_name]
        model = cfg["models"].get(tier, list(cfg["models"].values())[0])

        # 从环境变量覆盖模型名（支持 Ollama 等本地部署）
        if "model_env" in cfg:
            env_model = os.environ.get(cfg["model_env"], "")
            if env_model:
                model = env_model

        # 构建base_url
        if "base_url_env" in cfg:
            base_url = os.environ.get(cfg["base_url_env"], cfg.get("base_url", ""))
        else:
            base_url = cfg["base_url"]

        # Anthropic (Claude) 使用自有API格式
        if provider_name == "anthropic":
            return self._call_anthropic(
                api_key, model, system_prompt, user_prompt,
                temperature, max_tokens, cfg
            )

        # OpenAI兼容格式（DeepSeek/智谱/阿里/OpenAI均支持）
        if provider_name in ("deepseek", "zhipu", "alibaba", "openai_compatible"):
            return self._call_openai_compatible(
                base_url, api_key, model, provider_name,
                system_prompt, user_prompt, temperature, max_tokens, cfg
            )
        elif provider_name == "baidu":
            return self._call_baidu(
                api_key, model, system_prompt, user_prompt,
                temperature, max_tokens, cfg
            )

        raise ValueError(f"未支持的提供商: {provider_name}")

    def _call_openai_compatible(self, base_url, api_key, model, provider_name,
                                 system_prompt, user_prompt, temperature,
                                 max_tokens, cfg) -> LLMResponse:
        """OpenAI兼容API调用"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        elapsed = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        result = resp.json()
        choice = result["choices"][0]
        usage = result.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        cost = (in_tok * cfg["price_input"] + out_tok * cfg["price_output"]) / 1e6

        return LLMResponse(
            content=choice["message"]["content"],
            model=model,
            provider=provider_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_yuan=round(cost, 4),
            elapsed_ms=elapsed,
        )

    def _call_anthropic(self, api_key, model, system_prompt, user_prompt,
                        temperature, max_tokens, cfg) -> LLMResponse:
        """Anthropic Claude API调用"""
        url = f"{cfg['base_url']}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        elapsed = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        result = resp.json()
        content = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = result.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cost = (in_tok * cfg["price_input"] + out_tok * cfg["price_output"]) / 1e6

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_yuan=round(cost, 4),
            elapsed_ms=elapsed,
        )

    def _call_baidu(self, api_key, model, system_prompt, user_prompt,
                    temperature, max_tokens, cfg) -> LLMResponse:
        """百度千帆API调用（需要access_token）"""
        # 百度API需要先获取access_token
        # api_key格式: "client_id:client_secret" 或直接access_token
        if ":" in api_key:
            client_id, client_secret = api_key.split(":", 1)
            token_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
            token_resp = requests.post(token_url, timeout=10)
            access_token = token_resp.json().get("access_token")
        else:
            access_token = api_key

        url = f"{cfg['base_url']}/{model}?access_token={access_token}"
        payload = {
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ],
            "temperature": temperature,
        }

        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=180)
        elapsed = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        result = resp.json()
        usage = result.get("usage", {})

        return LLMResponse(
            content=result.get("result", ""),
            model=model,
            provider="baidu",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_yuan=0,
            elapsed_ms=elapsed,
        )

    @staticmethod
    def _min_tier(a: str, b: str) -> str:
        order = {"T1": 1, "T2": 2, "T3": 3}
        return a if order.get(a, 2) <= order.get(b, 3) else b
