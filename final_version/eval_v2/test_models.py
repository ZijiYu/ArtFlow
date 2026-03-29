import requests
import json

# --------------------------
# 配置
# --------------------------
API_URL = "https://api.zjuqx.cn/v1/embeddings"  # 嵌入接口
API_KEY = "sk-jGKJtrju4HnIdttWD902Ad017d2d484b93F0CcEc08CcA9A6"  # 替换成你的 API Key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 测试文本
TEST_TEXT = "这是一个嵌入测试文本"

# 要测试的模型列表
MODEL_LIST = [
    "baai/bge-m3",               # 原本不可用
    "text-embedding-3-small",    # 推荐
    "text-embedding-3-large",    # 推荐
    "text-embedding-ada-002",    # 旧版
    "BAAI/bge-base-en-v1.5"      # 可能可用
]

# --------------------------
# 测试函数
# --------------------------
def test_embedding_model(model_name):
    payload = {
        "model": model_name,
        "input": TEST_TEXT
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            embedding_len = len(data.get("data")[0].get("embedding", []))
            print(f"[✅] 模型可用: {model_name}, 嵌入长度: {embedding_len}")
        else:
            print(f"[❌] 模型不可用: {model_name}, 状态码: {response.status_code}, 返回: {response.text}")
    except Exception as e:
        print(f"[⚠️] 调用失败: {model_name}, 错误: {str(e)}")

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    print("开始测试嵌入模型可用性...\n")
    for model in MODEL_LIST:
        test_embedding_model(model)