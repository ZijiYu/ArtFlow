"""
测试配置加载
验证从 config.yaml 读取的 API 配置是否正确
"""
from config import OPENAI_CONFIG

print("=" * 60)
print("📋 API 配置加载测试")
print("=" * 60)

print(f"\n✅ Base URL: {OPENAI_CONFIG['base_url']}")
print(f"✅ Model: {OPENAI_CONFIG['model']}")
print(f"✅ API Key: {OPENAI_CONFIG['api_key'][:20]}...{OPENAI_CONFIG['api_key'][-4:]}")

print("\n" + "=" * 60)
print("配置加载成功！")
print("=" * 60)

# 验证模型是否为 openai/gpt-5.4
if OPENAI_CONFIG['model'] == 'openai/gpt-5.4':
    print("\n✅ 模型已设置为 openai/gpt-5.4")
else:
    print(f"\n⚠️  警告：当前模型为 {OPENAI_CONFIG['model']}，不是 openai/gpt-5.4")

# 验证 API Key 是否有效
if OPENAI_CONFIG['api_key'] and len(OPENAI_CONFIG['api_key']) > 20:
    print("✅ API Key 格式正确")
else:
    print("❌ API Key 无效或为空")

# 验证 Base URL
if "api.zjuqx.cn" in OPENAI_CONFIG['base_url']:
    print("✅ Base URL 正确")
else:
    print(f"⚠️  Base URL: {OPENAI_CONFIG['base_url']}")
