# GPU 候选环境

无卡安装复核：上游numpy override安装2.4.6，与mistral-common在Python3.12下的<2.4上限冲突。source修复模式现另行固定numpy2.3.5官方wheel（清单及SHA256见flash-source-repair.json），远端255包pip check通过。此项是明确的安装偏差，不是原上游锁完全原样安装。CUDA/FA真实验收仍待执行。

2026-09-28修复：上游flash-attn预编译wheel实测404。A100/A800安装改用bootstrap的`--flash-attn-source`，按本目录`flash-source-repair.json`下载校验官方2.8.3源码并编译sm80；其余包继续frozen sync，上游四个原文件不改。额外构建工具wheel固定0.45.1。真实Linux构建和GPU探针仍待验收；后续不要使用普通uv sync覆盖已修复环境。

来源：verl v0.9.1，commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`。本目录 uv.lock、pyproject.toml、setup.py、manage_envs.py、Dockerfile.uv.cu130 均是该提交的原文件；没有重新解析或改写上游锁。上游许可证随完整源归档提供（Apache-2.0）。

候选组合：Linux x86_64 / Python 3.12 / CUDA 13.0 / torch 2.11.0+cu130 / vLLM 0.24.0 / transformers 5.9.0 / PEFT 0.19.1 / Ray 2.55.1 / flash-attn 2.8.3。TransferQueue 与 mbridge 的 Git 提交由锁固定。`fsdp-vllm.requirements.txt` 由 uv 0.12.19 export --frozen --extra fsdp --extra vllm 导出。

**状态：来源和锁可验证；尚未在 GPU 上安装或验收。** CUDA 大版本满足 NVIDIA 对 R580 的要求不等于整个训练栈已验证。实际 import、BF16 kernel、7B 更新、vLLM 同步、log-prob 与恢复必须通过现场 pilot。

bootstrap 会逐文件对照已验 SHA256 的归档，拒绝修改/缺失/额外源码；新提取目录验收后原子发布。复用源码前删除其 __pycache__ 中的字节码文件，防止旧缓存覆盖真实源码；.venv 和 verl.egg-info 为构建产物，不属于上游源码证明。Windows source-only 会物化无法创建的内部 symlink 并校验副本内容；Linux 保留并校验原链接。旧的仅核对四个构建文件的验证已被替换。

付费授权后，在 Linux GPU 实例内执行：

```bash
python3 -m pip install uv==0.12.19
python3.12 scripts/bootstrap_gpu.py --runtime-root /root/autodl-tmp/posttrain-runtime --flash-attn-source
# 使用 install-receipt.json 中记录的 runtime_python
RUNTIME_PYTHON=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b/.venv/bin/python
"$RUNTIME_PYTHON" scripts/probe_gpu_environment.py --output runs/gpu-environment-v1.json
```

系统需要 Python ≥3.12（tarfile 安全提取过滤器）、git、g++、CUDA 13 开发环境及 >=65 GiB 可用数据盘。不要在 Windows 安装这组依赖。安装失败保留日志并停止计费任务，不自行替换版本；先记录兼容性问题及最小修订。

AutoDL 内部不支持 Docker；这里通过 frozen uv 环境安装，不执行本目录的上游 Dockerfile。执行判题仍在独立 CPU 虚机，不能移到训练进程。
