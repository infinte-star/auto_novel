# 通用 AI 写小说框架

一个内容无关的长篇中文网文自动生成流水线。给定一份「创作纲要」（`prompt.md`），
框架用一条**确定性流水线**自动循环：每 10 章规划一段弧、一次调用写完一章正文和状态、
确定性验收、能修就修，直到达到目标字数，并可选地做一遍分组精修（refine）。
路由分支全部是纯函数，没有自评分参与发布判定。

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
v2/run.py    v2/beat.py     v2/write.py   v2/accept.py   # 决策表 / 弧规划 / 写作 / 验收
v2/repair.py v2/canon.py    v2/anchor.py                 # 修复 / 状态投影 / 外部盲判
config.py    memory.py      writing.py    store.py       # 配置 / 记忆与 bootstrap / 写作教条 / 持久化
quality.py   fix.py         retrieval.py  checkpoint.py  llm.py
refine.py    package.py     screenplay.py                # 完稿后工具

```

---

## 工作原理

一条**确定性流水线**：每章的分支全部由纯函数决定，只有四类动作会调用模型
（规划一段弧、写一章、查设定、修复）。平均约 2.5 次调用/章。

1. **bootstrap**（首次运行）：读 `prompt.md`，生成 `state.md` 和
   `memory/{bible,characters,timeline,threads,volume_plan}.md`。
2. **主循环**：`find_last_chapter()` → `run_chapter()`，直到字数达标或到 `max_chapters`。
3. **每章流程**由 `v2/run.py` 的决策表驱动，每做完一个动作**从表头重新判一遍**：

   ```
   need_card   → 需要卡片？每 arc_span(10) 章一次弧规划，一次性产出整段的 ChapterCard
   card_invalid→ 卡片有硬伤？确定性校验，CRITICAL 才修
   need_draft  → 没有正文？一次调用同时产出正文和状态增量（无独立抽取调用）
   need_report → 没有验收报告？零 LLM，跑确定性门
   l0_pending  → L0 能修的（零 LLM：破折号/碎句/化石轮换/景物开场降级）
   l1_pending  → L1 能修的（有限次定点重写：扩写到字数带、补对白）
   canon_pending→ 查设定冲突，结论必须能在正文里字符串命中，否则丢弃
   next_card_patch → 用本章实际发生的事修补下一章卡片
   rescue      → 仍有硬阻塞才重写（修复永远排在它前面）
   commit      → 落盘：chapters/、book.md、结构化状态、state.md
   ```

4. **一把尺子**：能否发布只由 `quality.hard_block_reasons` 判定，没有自评分参与。
   `review_round0.json` 记录的是**未经任何修复的首稿**，这正是 `tools/fpy_prime.py`
   离线复算首过率时读的文件。
5. **不闩锁**：验收集里每一项都是「本章正文改一改就能变绿」的量；书级累计量只告警、
   永不阻塞（`GATE_SCOPES`）。
6. **断点续写**：每步都写 checkpoint 到 `logs/checkpoints/chNNNN/`，中断后重新 `run`
   会从断点继续，不重复消耗 token。
7. **熔断**：连续 `quality_breaker_consecutive`（默认 2）章带着未解决的硬阻塞落盘就停机
   交人判断——这种失败模式不是多花 token 能修的。
8. **精修（可选）**：`refine_after_complete: true` 时，完成后按 5 章一组重写，
   输出到 `chapters_refined/` 和 `book_refined.md`，原文不动。

---

## 质量护栏（防塌缩）

最大的失败模式是**文风塌缩**：正文逐渐退化成「句子——状态——状态」式的破折号碎句，
而模型自评因为自身文风也跟着漂移，反而给这种碎句打 9+ 分。所以引擎的发布判据里
**完全不含自评分**，全部是确定性的客观锚点（`quality.py` / `retrieval.py`，
由 `v2/accept.py` 组装成验收集）：

- **规则文体检测** `quality.py:style_health`：非 LLM 的确定性指标——破折号密度、
  平均句长、碎句行占比、对话有无。算出 `penalty`，超过 `style_penalty_block`
  直接拦截，并把整改指令注入下一章写作提示。
- **契约兑现校验（CCC）** `v2/accept.py:contract_fulfilment`：卡片承诺的地点/人物/
  转折关键物/结尾钩子/禁忌逐项在正文里核对。卡片字段按设计就是具体的，所以
  「有没有写到」是零 LLM 可判定的。
- **引用即丢弃（cite-or-drop）** `v2/accept.py:citation_check`：任何指不出正文原文
  片段的评审结论一律丢弃，而不是打个折扣计入——这是防评审凭空编造违规的唯一手段。
- **场景语义去重** `quality.py:scene_similarity`：新卡片骨架与近期卡片的 Jaccard
  相似度超过 `scene_dedupe_sim_warn` 时告警并追加硬约束，阻止「无限切片同一场景」。
- **全书化石** `quality.py:book_wide_fossils`：跨全书复现的 6 字 CJK n-gram
  （6 章窗口的重复检测结构性看不到的那一类）。
- **检索式记忆 (RAG)** `retrieval.py`：零额外依赖的 TF-IDF 字符二元组检索
  （不用 embedding，唯一依赖仍是 `openai`），把被摘要压缩掉的早期具体事实重新
  召回到写作上下文。索引在 `save_chapter` 时幂等写入 `logs/retrieval_index.json`。
- **固定文风基线** `memory/voice_baseline.md` 是**冻结**的：从漂移的正文里重新提炼
  文风，正是造成塌缩的自投喂回路。
- **外部盲判** `v2/anchor.py`：唯一读正文的指标。盲判（只标甲/乙）、双向判
  （交换位置两次一致才计胜）、**不带 cacheable_prefix**——这是引擎唯一无法自己
  颁给自己的分数。

> 这些护栏由 `config.yaml` 的开关控制，默认开启；阈值见下节。
> `logs/retrieval_index.json` 可安全删除，会自动重建。
> `memory/voice.md` **不要删**：它只在 bootstrap 生成一次，而 bootstrap 只在
> `state.md` 缺失时才重跑，所以删掉它不会重建——只会让之后每一章的提示词里
> 静默少掉「叙事声音」这一节。

---

## 配置要点（`novels/<名字>/config.yaml`）

配置用一个**精简版 YAML 子集**解析（只认 `section:` 和缩进的 `key: value`，
不支持嵌套/列表/锚点）。常调的几个：

**`novel:` 段**
- `engine` — 引擎版本，只认 `v2`（不写等于 `v2`）。旧配置里写着 `v1` 会**直接报错退出**，
  不会静默降级：两代引擎写的 checkpoint 标签和验收字段不同，猜错等于伪造测量。
- `target_words` — 目标总字数（达到即停）
- `chapter_words` — 单章目标字数
- `chapter_min_chars` — 单章字数下限（低于此值会触发 L1 定点扩写）
- `max_chapters` — 章节数硬上限（0 或不写 = 不限，仅按字数停）
- `arc_span` — 每多少章做一次弧规划（默认 10）
- `quality_breaker_consecutive` — 连续多少章带着未解决硬阻塞落盘就停机（默认 2）
- `style_preset` — 题材预设，驱动写作/评审/精调的题材化提示词：`history`（历史厚重）/ `xuanhuan_shuang`（穿越爽文）/ `system_stream`（系统流）/ `urban_ability`（都市异能·重生）/ `romance_female`（女频言情·宠文）/ `wanzu_xuanhuan`（现代玄幻·万族）/ `suspense`（悬疑惊悚）/ `rule_horror`（规则怪谈·民俗无限流，别名 `guize`/`infinite_flow`）
- `creative_boost_enabled` — bootstrap 阶段一次性 AI 创意增强（跨题材联想注入新颖金手指/人设梗/开篇钩子，默认开）
- `opening_chapters` — 开篇黄金三章特化章数（写作注入强钩子规则 + 开篇 hook 门更严）
- `opening_trial_variants` / `opening_trial_chapters` — `trial` 命令的默认开篇试写数量与每条路线章数
- `platform_preset` — 平台/读者画像：`general` / `qidian_male` / `fanqie_free` / `jinjiang_female` / `qimao_free`
- `benchmark_enabled` / `benchmark_dir` / `benchmark_top_k` — 本地爆款样本库召回，默认读取 `benchmarks/`
- `memory_*_chars` — 各记忆层读取上限，避免长期文件膨胀污染上下文
- `package_after_complete` / `refine_after_complete` — 是否完成后自动出书籍包装 / 自动精修（默认都关）

**质量护栏开关（`novel:` 段，默认开启）**
- `style_health_enabled` + `style_em_dash_per_kchar_warn/_bad`、
  `style_min_avg_sentence_chars`、`style_fragment_line_ratio_max`、
  `style_penalty_cap`、`style_penalty_block` — 规则文体检测与扣分/拦截阈值
- `scene_dedupe_enabled` / `scene_dedupe_sim_warn` / `_sim_block` / `_sim_identical` — 场景骨架去重的告警/重规划/绝对上限三档
- `book_fossil_enabled` / `book_fossil_chapter_frac` / `book_fossil_hard_ratio` — 全书化石短语检测
- `style_cross_repeat_reject_count` — 跨章原句复用的拒收计数
- `descriptor_frequency_enabled` / `genre_drift_reject_enabled` / `opening_hook_gate_enabled` — 用词频次 / 题材漂移 / 开篇钩子门
- `dialogue_health_enabled` / `dialogue_char_ratio_min` / `_target` — 对白占比门（由 L1 补对白应答）
- `rag_enabled` / `rag_top_k` / `rag_exclude_recent` — 检索式记忆
- `fix_max_l1_calls` — L1 定点修复的调用上限
- `fossil_tail_anchor_enabled` — 把硬化石禁令再钉一遍到写作提示词末尾

**爆款样本库**
- 样本放到 `benchmarks/<platform>/<style>/`，支持 `.md` / `.txt` / `.json`
- 只放结构摘要、开篇模式、兑现节奏、禁忌，不要放未经授权的整章正文
- 生成大纲/正文时会自动召回相近样本，作为结构参照而非文本模仿
- 可用 `python novel.py benchmark add qidian_male history path/to/sample.json` 导入结构化样本

**读者承诺账本**
- 状态增量会把 `thread_type: reader_promise` 同步到独立账本
- 后续弧规划会看到活跃/逾期承诺，减少“只开钩子不兑现”

**开篇路线采纳**
- `trial` 只试写，不污染正式正文
- `adopt-trial` 会把最佳路线写到 `memory/opening_route.md`
- ⚠️ **当前 v2 引擎还没读这个文件**（它自己做上下文投影），所以 `adopt-trial` 暂时
  只对 `trial` / `package` 这类走旧记忆层的命令生效。待修。

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
- 所有小说一律通过 `novel.py` 管理，产物收纳在 `novels/<名字>/` 下，不污染根目录。
