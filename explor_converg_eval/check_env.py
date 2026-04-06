"""
环境检查脚本：验证所有配置是否正确
"""
import os
import sys
from pathlib import Path

print("="*60)
print("🔍 环境检查")
print("="*60)

# 1. Python版本
print(f"\n✓ Python版本: {sys.version.split()[0]}")

# 2. 工作目录
print(f"✓ 工作目录: {Path.cwd()}")

# 3. API密钥
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key:
    print(f"✓ OPENAI_API_KEY: {api_key[:10]}...{api_key[-4:]} (已设置)")
else:
    print("❌ OPENAI_API_KEY: 未设置")
    print("   请运行: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

# 4. 输入文件
from config import GT_JSONL_PATH, ZHIHUA_TXT_DIR

if GT_JSONL_PATH.exists():
    print(f"✓ GT文件: {GT_JSONL_PATH} (存在)")
else:
    print(f"❌ GT文件: {GT_JSONL_PATH} (不存在)")
    sys.exit(1)

if ZHIHUA_TXT_DIR.exists():
    txt_count = len(list(ZHIHUA_TXT_DIR.glob("*.txt")))
    print(f"✓ zhihua目录: {ZHIHUA_TXT_DIR} ({txt_count}个txt文件)")
else:
    print(f"❌ zhihua目录: {ZHIHUA_TXT_DIR} (不存在)")
    sys.exit(1)

# 5. 依赖检查
print(f"\n检查依赖包...")
required_packages = [
    'aiohttp', 'openai', 'instructor', 'pydantic',
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tqdm'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} (未安装)")
        missing.append(pkg)

if missing:
    print(f"\n需要安装缺失的包:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

# 6. result目录
from config import RESULT_DIR, EXTRACTED_DIR, METRICS_DIR, REPORTS_DIR, LOGS_DIR

print(f"\nresult目录结构:")
for dir_path in [EXTRACTED_DIR, METRICS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_path.relative_to(Path.cwd())}")

print("\n" + "="*60)
print("✅ 环境检查完成！可以开始运行")
print("="*60)
print(f"\n运行命令:")
print(f"  python incremental_runner.py")
