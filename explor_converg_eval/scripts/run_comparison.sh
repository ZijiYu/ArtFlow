#!/bin/bash
# 多数据源对比分析启动脚本

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

cd "$PROJECT_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
    echo -e "${YELLOW}未检测到本地虚拟环境，正在创建 .venv ...${NC}"
    python3 -m venv "$VENV_DIR" || {
        echo -e "${RED}❌ 创建虚拟环境失败，请先检查 python3 是否可用${NC}"
        exit 1
    }
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  多数据源对比分析${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}📋 API 配置来源: /Users/ken/MM/Pipeline/final_version/config.yaml${NC}"

echo -e "\n${BLUE}检查依赖...${NC}"
"$PYTHON_BIN" -c "import instructor, openai, pydantic, numpy, pandas, yaml" 2>/dev/null
DEPS_CHECK=$?

if [ "$DEPS_CHECK" -ne 0 ]; then
    echo -e "${YELLOW}⚠️  检测到缺少依赖，开始安装 requirements.txt ...${NC}"
    "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt" || {
        echo -e "${RED}❌ 依赖安装失败${NC}"
        echo -e "${YELLOW}请尝试手动运行: bash scripts/install_deps.sh${NC}"
        exit 1
    }
fi

echo -e "${GREEN}✅ 依赖检查完成${NC}"

echo -e "\n${BLUE}API 配置信息:${NC}"
"$PYTHON_BIN" - <<'PY'
from config import OPENAI_CONFIG

api_key = OPENAI_CONFIG["api_key"]
masked_key = f"{api_key[:20]}..." if api_key else "(empty)"

print(f"  Base URL: {OPENAI_CONFIG['base_url']}")
print(f"  Model: {OPENAI_CONFIG['model']}")
print(f"  API Key: {masked_key}")
PY

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 配置加载失败${NC}"
    echo -e "${YELLOW}请检查: $PYTHON_BIN test_config.py${NC}"
    exit 1
fi

MODE=${1:-test}
COMPARE_SCOPE=${2:-intersection}
EXTRA_ARGS=("${@:3}")

echo -e "\n${BLUE}开始运行对比分析...${NC}"
echo -e "${YELLOW}模式: $MODE${NC}"
echo -e "${YELLOW}样本范围: $COMPARE_SCOPE${NC}"
echo ""

if [ "$MODE" = "test" ]; then
    echo -e "${YELLOW}📝 测试模式：每个数据源处理前10条${NC}"
    RUN_CMD=( "$PYTHON_BIN" "$PROJECT_ROOT/compare_three_sources.py" --mode test --compare-scope "$COMPARE_SCOPE" "${EXTRA_ARGS[@]}" )
elif [ "$MODE" = "all" ]; then
    echo -e "${YELLOW}🚀 完整模式：处理所有数据${NC}"
    echo -e "${YELLOW}⏱️  预计耗时：10-15分钟${NC}"
    RUN_CMD=( "$PYTHON_BIN" "$PROJECT_ROOT/compare_three_sources.py" --mode all --compare-scope "$COMPARE_SCOPE" "${EXTRA_ARGS[@]}" )
else
    echo -e "${RED}❌ 未知模式: $MODE${NC}"
    echo -e "${YELLOW}用法: bash scripts/run_comparison.sh [test|all] [intersection|union] [额外参数]${NC}"
    exit 1
fi

if "${RUN_CMD[@]}"; then
    echo -e "\n${GREEN}✅ 对比分析完成！${NC}"
    echo -e "${BLUE}结果保存在: result/comparison/${NC}"
    echo -e "${BLUE}报告保存在: result/comparison/reports/${NC}"
else
    echo -e "\n${RED}❌ 对比分析失败${NC}"
    echo -e "${YELLOW}请先检查控制台中的 API / 数据 / 配置报错${NC}"
    exit 1
fi
