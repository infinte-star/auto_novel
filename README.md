# 通用 AI 写小说框架

一个内容无关的长篇中文网文自动生成流水线。给定一份「创作纲要」（`prompt.md`），
框架自动循环执行 **规划 → 写作 → 评审 → 修订 → 抽取记忆**，直到达到目标字数，
并可选地做一遍分组精修（refine）。

支持**同时写多篇小说**：每篇小说有独立目录、独立配置、独立进程，互不干扰。

---

## 快速开始

```bash
# 1. 安装依赖（唯一依赖是 openai>=1.0.0）
pip install -r requirements.txt

# 2. 新建一篇小说（在 novels/<名字>/ 下生成 config.yaml 和 prompt.md）
python novel.py create 我的小说

# 3. 编辑创作纲要，填写类型/主角/世界观/卷纲等
#    novels/我的小说/prompt.md
#    （可选）按需调整 novels/我的小说/config.yaml 里的 target_words、chapter_words 等

# 4. 运行（后台独立进程，日志写到 novels/我的小说/logs/run.log）
python novel.py run 我的小说

# 5. 查看进度
python novel.py list
```

---

## `novel.py` 命令

| 命令 | 说明 |
| --- | --- |
| `python novel.py create <名字>` | 从模板创建 `novels/<名字>/`，含 `config.yaml`（路径已自动指向本目录）和待填写的 `prompt.md` |
| `python novel.py trial <名字>` | 生成多条开篇试写路线（默认 3 条 × 3 章），输出到 `logs/opening_trials/`，不污染正式 `chapters/` / `book.md` |
| `python novel.py adopt-trial <名字> [trial_id]` | 采纳某次 trial 的最佳开篇路线，写入 `memory/opening_route.md`，正式生成时优先遵循 |
| `python novel.py benchmark list/add ...` | 管理本地爆款样本库，样本用于结构召回，不复制正文 |
| `python novel.py run <名字>` | 后台分离进程运行流水线，自动从上次断点续写 |
| `python novel.py run <名字> --foreground` | 前台运行（attach 当前控制台，便于调试） |
| `python novel.py list` | 列出所有小说：章节数 / 字数 / 是否在跑 / 最新日志行 |
| `python novel.py stop <名字>` | 只杀这一篇的进程（按命令行 `run <名字>` token 精确匹配，不误伤其它小说） |
| `python novel.py restart <名字>` | 停止并重启（从断点续写） |

可同时 `run` 多篇小说，它们是各自独立的进程。

---

## 目录结构

```
novels/
  <小说名>/
    prompt.md            # 创作纲要（你填写）
    config.yaml          # 该小说配置，paths 全部指向本目录
    book.md              # 全书（自动拼接生成）
    state.md             # 当前状态摘要
    chapters/            # 每章 0001.md, 0002.md ...
    memory/              # bible/characters/timeline/threads/volume_plan/voice(s)
    logs/                # run.log, checkpoints/, refine/, memory_archive/, retrieval_index.json
    story_state.db       # SQLite 结构化状态（WAL）

config_template.yaml     # 新建小说的配置模板（含 __NOVEL__ 占位符）
prompt_template.md       # 创作纲要骨架模板
novel.py                 # 多小说统一 CLI 入口

# 核心引擎（内容无关，所有小说共用）
pipeline.py  config.py  memory.py  planning.py  writing.py
review.py    refine.py  store.py   checkpoint.py  llm.py
quality.py   retrieval.py            # 质量护栏：规则文体检测 / 场景去重 / 检索式记忆

# 旧版根目录长篇入口（向后兼容，操作根目录的 prompt.md/config.yaml）
run.py  restart.py  start_pipeline.bat  restart.bat
```

---

## 工作原理

1. **bootstrap**（首次运行）：读 `prompt.md`，用一次 LLM 调用生成
   `state.md` 和 `memory/{bible,characters,timeline,threads,volume_plan}.md`。
2. **主循环**：`find_last_chapter()` → `generate_one_chapter()`，直到字数达标。
3. **每章流程**：
   `规划候选方案(多策略 bandit) → 连续性校验 → 写作(可多候选) →
    评审/修订循环 → 弱结尾 hook 微调 → 保存 → 抽取事件 → 更新结构化状态/state.md`。
4. **质量护栏**（防文风塌缩 / 防自评分虚高，详见下节）：规则文体检测、
   独立冷读者、宏观推进度量、检索式记忆、场景去重、固定文风基线。
5. **后台任务**：事件抽取、阶段评审、记忆压缩、自适应重规划、下一章方案预取
   都在后台线程池跑，不阻塞关键路径。
6. **断点续写**：每个阶段都写 checkpoint 到 `logs/checkpoints/chNNNN/`，
   中断后重新 `run` 会从断点继续，不重复消耗 token。
7. **精修（可选）**：`refine_after_complete: true` 时，完成后按 5 章一组
   重写，输出到 `chapters_refined/` 和 `book_refined.md`，原文不动。

---

## 质量护栏（防塌缩）

最大的失败模式是**文风塌缩**：正文逐渐退化成「句子——状态——状态」式的破折号碎句，
而模型自评因为自身文风也跟着漂移，反而给这种碎句打 9+ 分。下面这些层就是专门
针对「LLM 自评不可信」设计的客观锚点（核心在 `quality.py` / `retrieval.py`，
并接入 `review.py` / `planning.py`）：

- **规则文体检测** `quality.py:style_health`：非 LLM 的确定性指标——破折号密度、
  平均句长、碎句行占比、对话有无。算出 `penalty` 直接从评审分里扣，超过
  `style_penalty_block` 直接拦截通过，并把整改指令注入下一章写作提示。
- **场景语义去重** `quality.py:scene_similarity`：新方案骨架与近期已选方案的
  Jaccard 相似度超过 `scene_dedupe_sim_warn` 时告警并追加硬约束，阻止「无限切片
  同一场景」。
- **检索式记忆 (RAG)** `retrieval.py`：零额外依赖的 TF-IDF 字符二元组检索
  （不用 embedding，唯一依赖仍是 `openai`），把被摘要压缩掉的早期具体事实重新
  召回到写作上下文。索引在 `save_chapter` 时幂等写入 `logs/retrieval_index.json`。
- **独立冷读者** `review.py:cold_reader_review`：每 `cold_reader_every` 章跑一次，
  **故意不带 cacheable_prefix**，因此不会像主评审那样被漂移的上下文「同化」而放水。
- **宏观推进度量** `review.py:macro_progress_check`：每 `macro_progress_every` 章
  对照 `volume_plan` 大事件锚点检测剧情停滞，停滞超阈值就写入加速指令。
- **固定文风基线** `review.py:refresh_voice_anchors`：锚定首次生成的
  `memory/voice_baseline.md`，正文出现塌缩迹象时**直接跳过 voice 刷新**，
  切断「劣化正文反过来成为新文风」的自投喂回路。
- **自适应降档** `planning.py`：质量长期稳定（窗口内最低分 ≥ `adaptive_downshift_score`）
  时自动减少候选方案数，省 token。

> 这些护栏由 `config.yaml` 的开关控制，默认开启；阈值见下节。
> `logs/retrieval_index.json` 和 `memory/voice_baseline.md` 都可安全删除，会自动重建。

---

## 配置要点（`novels/<名字>/config.yaml`）

配置用一个**精简版 YAML 子集**解析（只认 `section:` 和缩进的 `key: value`，
不支持嵌套/列表/锚点）。常调的几个：

**`novel:` 段**
- `target_words` — 目标总字数（达到即停）
- `chapter_words` — 单章目标字数
- `max_chapters` — 章节数硬上限（0 或不写 = 不限，仅按字数停）
- `quality_threshold` — 章节质量分阈值（评审低于此分会触发修订）
- `candidate_plans` / `candidate_chapters` — 并行候选方案/草稿数（择优）
- `style_preset` — 题材预设，驱动写作/评审/精调的题材化提示词：`history`（历史厚重）/ `xuanhuan_shuang`（穿越爽文）/ `system_stream`（系统流）/ `urban_ability`（都市异能·重生）/ `romance_female`（女频言情·宠文）/ `wanzu_xuanhuan`（现代玄幻·万族）
- `creative_boost_enabled` — bootstrap 阶段一次性 AI 创意增强（跨题材联想注入新颖金手指/人设梗/开篇钩子，默认开）
- `opening_chapters` / `opening_hook_strength_min` / `opening_review_strict` — 开篇黄金三章特化（写作注入强钩子规则 + 更高 hook 阈值 + 更严评审）
- `opening_trial_variants` / `opening_trial_chapters` — `trial` 命令的默认开篇试写数量与每条路线章数
- `replan_on_low_quality` — 章节低于阈值时，先重做大纲并重写一次，而不是只补丁修到勉强通过
- `platform_preset` — 平台/读者画像：`general` / `qidian_male` / `fanqie_free` / `jinjiang_female` / `qimao_free`
- `benchmark_enabled` / `benchmark_dir` / `benchmark_top_k` — 本地爆款样本库召回，默认读取 `benchmarks/`
- `pack_review_enabled` / `pack_review_every` — 10章包追读审校，检查承诺账本、爽点兑现、重复疲劳
- `pack_review_barrier` / `stage_review_barrier` — 上一包/上一阶段审校未完成时，下一章规划前等待其完成
- `memory_*_chars` — 各记忆层读取上限，避免长期文件膨胀污染上下文
- `refine_after_complete` — 是否完成后自动精修

**质量护栏开关（`novel:` 段，默认开启）**
- `style_health_enabled` + `style_em_dash_per_kchar_warn/_bad`、
  `style_min_avg_sentence_chars`、`style_fragment_line_ratio_max`、
  `style_penalty_cap`、`style_penalty_block` — 规则文体检测与扣分/拦截阈值
- `cold_reader_enabled` / `cold_reader_every` — 独立冷读者频率
- `macro_progress_enabled` / `macro_progress_every` / `macro_progress_stall_threshold` — 宏观推进度量
- `scene_dedupe_enabled` / `scene_dedupe_sim_warn` — 场景去重相似度阈值
- `scene_dedupe_sim_block` / `scene_dedupe_force_retry` — 场景骨架高度重复时硬性触发重规划
- `rag_enabled` / `rag_top_k` / `rag_exclude_recent` — 检索式记忆
- `voice_refresh_skip_penalty` — 检出塌缩时跳过 voice 刷新的扣分阈值
- `adaptive_downshift_enabled` / `_score` / `_window` / `_warmup` — 自适应降档

**爆款样本库**
- 样本放到 `benchmarks/<platform>/<style>/`，支持 `.md` / `.txt` / `.json`
- 只放结构摘要、开篇模式、兑现节奏、禁忌，不要放未经授权的整章正文
- 生成大纲/正文时会自动召回相近样本，作为结构参照而非文本模仿
- 可用 `python novel.py benchmark add qidian_male history path/to/sample.json` 导入结构化样本

**读者承诺账本**
- 事件抽取会把 `thread_type: reader_promise` 同步到独立账本
- 后续规划会看到活跃/逾期承诺，减少“只开钩子不兑现”
- 每 `pack_review_every` 章会做一次包评审，输出到 `logs/pack_reviews.md`

**开篇路线采纳**
- `trial` 只试写，不污染正式正文
- `adopt-trial` 会把最佳路线写到 `memory/opening_route.md`
- 后续规划/写作会把该路线作为高优先级约束，并自动刷新 prompt cache

**`api:` 段**
- `base_url` / `model` — 端点与模型
- `api_key` — 主 key；`api_keys` — 逗号/分号分隔的更多 key（同一 base_url 轮询）
- `api_key_groups` — `base_url|key1,key2;base_url2|key3` 形式的备用端点组
  （主 key 全挂时才回退）

> ⚠️ **同时运行多篇小说会共享同一批 API key 的 RPM/TPM 配额。**
> 想隔离配额，给不同小说的 config 配不同的 `api_key` / `api_key_groups`。

---

## 注意事项

- **`config_template.yaml` 内含真实 API key**：已被 `.gitignore` 忽略，但请勿
  把它或生成的 `novels/*/config.yaml` 提交到公开仓库。`create` 命令依赖该模板
  文件存在于磁盘上，不要删除。
- **后台启动优先用项目 venv** `E:\pycharmproject\allvenv\novel\Scripts\python.exe`
  （内含 `openai`）。可用 `NOVEL_PYTHON` 环境变量覆盖解释器路径。
- **进程隔离靠的是独立进程**：引擎里有进程级全局状态（prompt 路径、prompt 缓存），
  所以多篇并行用「每篇一个进程」而非单进程多线程。
- 旧版根目录长篇（`run.py`）和已迁移到 `novels/扶苏/` 的短篇互不影响。

---

## 旧版兼容

根目录的 `run.py` / `restart.py` 仍可运行那篇明末长篇（操作根目录的
`prompt.md` / `config.yaml` / `chapters/` / `book.md`）。新小说推荐一律走
`novel.py`，产物收纳在 `novels/<名字>/` 下，不污染根目录。
