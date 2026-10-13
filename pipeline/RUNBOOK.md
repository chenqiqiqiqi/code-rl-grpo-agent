# Stage 4 第一阶段操作手册（GPU 会话）

预登记见 `EXPERIMENT_PROTOCOL.md`（v4 补充：Stage 4 第一阶段预登记），方案见 `STAGE4_PHASE1_PLAN_DRAFT.md`。
约定：GPU 开机前先做完所有 VM/本地准备；GPU 任务结束立即提醒关机。

## A. 开 GPU 之前（本地 + VM，不花 GPU 钱）

1. 启动 VM（PowerShell）：`powershell -ExecutionPolicy Bypass -File scripts\start_local_vm.ps1 -Accelerator whpx -CpuCount 8 -MemoryMiB 5120`
2. VM 上重新签发隔离证明（24 小时有效，覆盖冒烟 + 二值分支及其恢复）：
   `cd /root/agent-cpu-local-20260929 && PYTHONPATH=src python3 -m posttrain isolation-check --image sha256:0106a6090782e770c43597f20b3e579584c575b1166d3a5dad969f18d4fe5e56 --output runs/isolation-stage4-<TAG>`
3. 复制 `stage4-phase1/vm_start_scorer.sh` 到 VM 为 `claude_start_scorer_stage4.sh`，启动：`bash claude_start_scorer_stage4.sh runs/isolation-stage4-<TAG>/attestation.json <TAG>`。确认没有其他判分任务（`claude-runner` 队列为空）。
4. 本机后台运行 `bash stage4-phase1/tunnels.sh`（两段隧道，断开自动重连）。

## B. GPU 开机后

1. **同步代码**：先比对 GPU 与本地 `src/` 的逐文件 SHA256，差异应只有本次改动的 `code_rl.py`、`generate.py`；确认后上传 `src/ scripts/ configs/single_rl_*_v4.json data/prepared/rl-pool-v5b/ EXPERIMENT_PROTOCOL.md STAGE4_PHASE1_PLAN_DRAFT.md stage4-phase1/ tests/test_stage4_phase1.py`，核对 `reports/verl-4090d-20260930/TASK4_VERL_4090D_ACCEPTANCE.md` 哈希与本地一致。
2. **GPU 端测试**：`pytest tests/test_stage4_phase1.py tests/test_agent_integration.py tests/test_single_model_rl.py tests/test_single_gpu_lora_sync.py tests/test_verl_pilot_config.py -q`。
3. **磁盘**：`df -h /root/autodl-tmp /`。只存 LoRA 后每个检查点约 1 GB，需求很小；旧的 15.7 GB 检查点列出清单交用户决定是否删除。
4. **证据目录** `evidence/stage4-<TAG>/`：`isolation/attestation.json`（来自 VM 的 `runs/isolation-stage4-<TAG>/`）、`gateway_identity.json`（来自 VM 的 `runs/scorer-stage4-<TAG>/gateway_identity.json`）；GPU 上 `curl --noproxy '*' 127.0.0.1:8080/identity` 必须与之相同。
5. **B1**：`python stage4-phase1/b1_from_pilot.py runs/verl-4090d-20260930/single`。若 `adjusted=true`，把 binary/mixed 配置的 `token_cap` 改为输出的 B1（两份一致），本地同步修改。
6. **freeze**（GPU 上生成，路径以 GPU 项目根为准）：
   `python scripts/make_rl_freeze.py --config configs/single_rl_<KIND>_v4.json --spent-cny 400 --forecast-cny 625 --output runs/stage4-phase1/freeze-<KIND>.json`，KIND = smoke / binary / mixed。

## C. 冒烟测试（约 30–45 分钟）

- `setsid nohup bash stage4-phase1/run_verl_formal.sh smoke configs/single_rl_smoke_v4.json runs/stage4-phase1/freeze-smoke.json evidence/stage4-<TAG> > /dev/null 2>&1 &`
- 第 2 步检查点出现后，在第 3 步生成/判分期间断开 GPU 侧隧道约 60 秒，确认训练按"无数值奖励即中止"退出；恢复隧道后用最新 receipt 恢复：同一命令末尾加 `runs/stage4-phase1/smoke/audit/checkpoints/<session>-<step>.json`。
- 通过条件：恢复会话从下一步开始；预算用尽后写出 `audit/completion.json`（`normal_token_budget_end`），最终检查点为最后一次完整更新且只含 LoRA；被中断那一步保留最坏预留；`max_num_seqs` 16 下显存峰值 < 47 GB；`monitor.py` 无告警。
- 显存不够时改回 `max_num_seqs` 8，重新生成三个 freeze 后重做冒烟。

## D. 二值分支（约 6 小时）

- `setsid nohup bash stage4-phase1/run_verl_formal.sh binary configs/single_rl_binary_v4.json runs/stage4-phase1/freeze-binary.json evidence/stage4-<TAG> > /dev/null 2>&1 &`
- 定期 `python stage4-phase1/monitor.py runs/stage4-phase1/binary`，按预登记中止规则处理；中断则按 C 的方式恢复。
- 结束后：
  1. 导出终点 adapter 和倒数第二个检查点 k 的 adapter：`python scripts/export_rl_adapter.py --run-dir runs/stage4-phase1/binary --receipt <receipt> --model-snapshot /root/autodl-tmp/models/qwen7b --output runs/stage4-phase1/binary-export-<step>`；
  2. 导出核对：`python scripts/check_rl_adapter_logprobs.py --export runs/stage4-phase1/binary-export-<k> --run-dir runs/stage4-phase1/binary --model-snapshot /root/autodl-tmp/models/qwen7b --output runs/stage4-phase1/binary-export-check.json`，要求 `passed` 且 `adapter_effect_detectable`；
  3. 开发集生成：配置取 M0 基线 seed 42 生成目录 `identity.json` 里的 `configuration`，原样写成 `stage4-phase1/gen_dev_k8_s42.json`；`python -m posttrain.training.generate --config stage4-phase1/gen_dev_k8_s42.json --prompts data/prepared/rl-pool-v5b/single-development.jsonl --output runs/stage4-phase1/binary-dev-s42 --model-snapshot /root/autodl-tmp/models/qwen7b --rl-adapter runs/stage4-phase1/binary-export-<final>`。判分等混合分支结束后再做（避免与训练争用 VM）。

## E. 混合分支（约 11 小时）

- 先重新签发隔离证明并重启网关（A.2–A.3，新 TAG），更新 `evidence/stage4-<TAG2>/`；freeze 不含证明，无需重做。
- 同 D，NAME=mixed，配置/freeze 换成 mixed。结束后同样导出、核对、生成开发集候选。
- **GPU 此时空闲 → 提醒用户关机。**

## F. 判分与选择（VM + 本地）

- 两份开发集候选用 `strip_cands.py` 处理后放入 VM 队列 `claude_score_train.py <cands> <out> --dev`（每份约 3 分钟）。
- 按预登记规则计算 Δ；|Δ| ≤ 2 时开 GPU 补 seed 43 生成（每份约 20 分钟），再判分。
- 写第一阶段报告，确定 W。
