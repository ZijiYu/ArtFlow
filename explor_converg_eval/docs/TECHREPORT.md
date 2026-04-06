# Explor Converg Eval 技术报告（TECHREPORT）

## 1. 项目目标
`explor_converg_eval` 用于对多数据源文本进行统一评估，核心回答两个问题：
- 广度（Diversity）：是否覆盖更多、更均衡的知识维度
- 深度（Depth）：单位文本长度内是否包含更多高价值信息

系统最终输出样本级指标、跨源对比表、图表与报告索引，支持测试模式快速验证与全量模式正式评估。

---

## 2. 测试逻辑（评估流程）

## 2.1 入口与运行模式
入口脚本：`compare_three_sources.py`（可由 `scripts/run_comparison.sh` 调用）。

关键参数：
- `--mode {test|all}`
  - `test`：快速回归验证
  - `all`：全量正式评估
- `--compare-scope {intersection|union}`
  - `intersection`：仅比较多个数据源共同存在的 `image_id`
  - `union`：每个数据源独立统计全部样本
- `--source`：可重复定义数据源
- `--gt-source`：指定作为对照基线的 GT 源名

`test` 模式下采样逻辑：
- `intersection`：公共 `image_id` 中取前 10 个，再对每个源取同一批样本
- `union`：每个源各取前 10 条

---

## 2.2 执行阶段（从输入到报告）
一次完整评估分为 5 步：

1) **解析数据源**
- 支持 `jsonl` 与 `txt_dir`
- `jsonl` 读取 `id_field`、`text_field`
- `txt_dir` 读取 `*.txt` / `*.md`，文件名 stem 作为 `image_id`

2) **样本对齐与抽样**
- 构建每个源的 `image_id -> record` 索引
- 去除同源重复 `image_id`（保留首次出现）
- 按 `compare_scope` 形成最终待处理样本集合

3) **结构化抽取（模块2）**
- 对每条文本抽取维度化 slot（关键词、相关性、权重等）
- 输出 `extracted_*.jsonl`

4) **指标计算（模块3）**
- 对每条样本计算广度/深度/相关性指标
- 若配置 `gt_source`，追加与 GT 的差分指标
- 输出 `comparison_*.csv`

5) **可视化与报告导出（模块4）**
- 输出散点图、柱状图、箱线图、权重分布、维度矩阵热力图
- 输出汇总表、术语表及 `report_summary.md`

---

## 3. 指标定义与计算公式

## 3.1 符号约定
- 维度集合：`D = {d1, d2, ..., dK}`，其中 `K=15`
- `n_d`：维度 `d` 的有效 slot 数（仅统计强相关+弱相关）
- `N = Σ_d n_d`：总有效 slot 数
- `L`：清洗后文本长度（字符数）
- 第 `i` 个有效 slot 的权重为 `w_i ∈ {1,2,3}`
- `W = Σ_i w_i`：总权重
- `N_s, N_w, N_ir`：强相关、弱相关、不相关 slot 数

---

## 3.2 广度相关指标

1) **维度覆盖数**
$$
\text{dimension\_coverage}=\sum_{d \in D}\mathbf{1}[n_d>0]
$$
含义：覆盖了多少个维度，取值 `0~15`。值越大，覆盖越广。

2) **熵（Diversity）**
$$
p_d=\frac{n_d}{N},\quad
\text{Entropy}=-\sum_{d \in D, n_d>0} p_d\log_2 p_d
$$
含义：维度分布越均衡，Entropy 越高；越集中在少数维度，Entropy 越低。

---

## 3.3 深度相关指标

1) **传统密度**
$$
\text{Density}=\frac{N}{L}
$$
含义：单位长度里的有效信息点数量。

2) **加权密度**
$$
\text{Weighted\_Density}=\frac{\sum_{i=1}^{N}w_i}{L}=\frac{W}{L}
$$
含义：单位长度里的“加权信息量”，相比 `Density` 更强调高权重信息。

3) **权重统计**
$$
\text{Avg\_Weight}=\frac{W}{N}\quad(N>0)
$$
并统计 `weight_distribution={1:count,2:count,3:count}`。

---

## 3.4 相关性质量指标

令：
$$
T=N_s+N_w+N_{ir}
$$
则：
$$
\text{strong\_relevant\_ratio}=\frac{N_s}{T},\quad
\text{weak\_relevant\_ratio}=\frac{N_w}{T},\quad
\text{irrelevant\_ratio}=\frac{N_{ir}}{T}
$$
含义：描述抽取结果的相关性构成。

---

## 3.5 与 GT 的差分指标
当样本命中同 `image_id` 的 GT 记录时，计算：

1) **绝对差分**
$$
\text{entropy\_diff\_from\_gt}=|\text{Entropy}_{model}-\text{Entropy}_{gt}|
$$
$$
\text{density\_diff\_from\_gt}=|\text{Density}_{model}-\text{Density}_{gt}|
$$
$$
\text{weighted\_density\_diff\_from\_gt}=|\text{WDensity}_{model}-\text{WDensity}_{gt}|
$$

2) **维度分布距离（余弦距离）**
- 向量构造：\(\mathbf{m}=[n_{d1},...,n_{dK}],\mathbf{g}=[n'_{d1},...,n'_{dK}]\)
- 余弦相似度：
$$
\cos(\theta)=\frac{\mathbf{m}\cdot\mathbf{g}}{\|\mathbf{m}\|\|\mathbf{g}\|}
$$
- 距离定义：
$$
\text{dimension\_distance\_from\_gt}=1-\cos(\theta)
$$

解释：
- 越接近 0：维度分布方向越接近 GT
- 越大：分布偏离越明显
- 若任一向量全 0（无有效维度信息），实现中返回 `1.0` 作为最大差异

---

## 4. 输入与输出

## 4.1 输入
- JSONL 数据源（需配置 `id_field` 与 `text_field`）
- 文本目录数据源（`*.txt` / `*.md`）

## 4.2 输出
核心输出目录：`/Users/ken/MM/Pipeline/explor_converg_eval/result/comparison/`
- `comparison_*.csv`：样本级指标明细
- `extracted_*.jsonl`：结构化抽取结果
- `reports/<run_name>/`：图表、汇总表、术语统计、`report_summary.md`

---

## 5. 快速使用

测试模式（推荐先跑）：
```bash
cd /Users/ken/MM/Pipeline/explor_converg_eval
bash scripts/run_comparison.sh test intersection
```

全量模式：
```bash
cd /Users/ken/MM/Pipeline/explor_converg_eval
bash run_eval.sh
```

---

## 6. 方法学注意事项
- `dimension_coverage` 是“覆盖广度”，不是“与 GT 对齐度”。
- `dimension_distance_from_gt` 是“分布方向差异”，对同向同比例放大不敏感。
- 公平横向比较优先使用 `intersection`。
- 各源独立画像分析可使用 `union`。
