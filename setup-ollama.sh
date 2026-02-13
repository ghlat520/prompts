#!/bin/bash
# ============================================================
# Ollama + qwen2.5:7b 一键安装脚本
# 适用: macOS (Apple Silicon / Intel)
# 作用: 安装 Ollama → 拉取 qwen2.5:7b → 配置 .env → 验证
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
MODEL="qwen2.5:14b"

echo "=============================="
echo " Ollama 一键安装 (股票分析专用)"
echo "=============================="
echo ""

# ── 1. 检查是否已安装 ──
if command -v ollama &>/dev/null; then
    echo "[✓] Ollama 已安装: $(ollama --version)"
else
    echo "[1/4] 下载并安装 Ollama..."
    # macOS: 下载官方安装包
    TMPFILE="/tmp/Ollama-darwin.zip"
    curl -fsSL -o "$TMPFILE" "https://ollama.com/download/Ollama-darwin.zip"
    echo "  解压安装中..."
    unzip -q -o "$TMPFILE" -d /Applications/
    rm -f "$TMPFILE"
    echo "[✓] Ollama 已安装到 /Applications/Ollama.app"
fi

# ── 2. 启动 Ollama 服务 ──
echo ""
echo "[2/4] 启动 Ollama 服务..."
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "[✓] Ollama 服务已在运行"
else
    open /Applications/Ollama.app
    echo "  等待服务启动..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            echo "[✓] Ollama 服务已启动"
            break
        fi
        sleep 1
    done
    # 最终检查
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        echo "[✗] Ollama 服务启动失败，请手动打开 Ollama.app 后重试"
        exit 1
    fi
fi

# ── 3. 拉取模型 ──
echo ""
echo "[3/4] 拉取模型 $MODEL (~9GB，首次下载需几分钟)..."
if ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo "[✓] 模型 $MODEL 已存在"
else
    ollama pull "$MODEL"
    echo "[✓] 模型 $MODEL 拉取完成"
fi

# ── 4. 配置 .env ──
echo ""
echo "[4/4] 配置 .env..."
# 追加或创建 .env，不覆盖已有 key
if [ -f "$ENV_FILE" ] && grep -q "OPENAI_BASE_URL" "$ENV_FILE"; then
    echo "[✓] .env 已配置，跳过"
else
    cat >> "$ENV_FILE" << 'EOF'

# Ollama 本地 LLM (自动配置)
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
EOF
    echo "[✓] 已写入 .env: OPENAI_API_KEY=ollama, OPENAI_BASE_URL=http://localhost:11434/v1"
fi

# ── 验证 ──
echo ""
echo "=============================="
echo " 验证安装"
echo "=============================="
echo ""
echo "Ollama 版本: $(ollama --version 2>/dev/null || echo 'N/A')"
echo "模型列表:"
ollama list 2>/dev/null || echo "  (无法获取)"
echo ""
echo "API 测试:"
RESP=$(curl -sf http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，请用一句话介绍自己\"}],\"max_tokens\":50}" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "[✓] LLM 响应正常"
    echo "  回复: $(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null || echo "$RESP")"
else
    echo "[✗] API 调用失败，请检查 Ollama 是否正常运行"
    exit 1
fi

echo ""
echo "=============================="
echo " 安装完成！"
echo "=============================="
echo ""
echo "启动股票分析服务:"
echo "  python3 $SCRIPT_DIR/server.py"
echo ""
echo "然后打开浏览器: http://localhost:8899"
echo ""
