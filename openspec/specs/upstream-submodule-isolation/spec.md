# upstream-submodule-isolation Specification

## Purpose

定义 `wekws` 上游依赖的只读边界和版本固定方式，保证项目可复现构建，同时避免本地修改或下载资源混入上游仓库。

## Requirements

### Requirement: wekws 是 pin 住的 git submodule

项目 SHALL 通过 git submodule 引入 `wekws`，并 pin 到 commit `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`；外层仓库 MUST 不提交 `wekws/` 的源码文件。

#### Scenario: 全新 checkout

- **WHEN** 使用者初始化外层仓库并更新 submodule
- **THEN** `wekws/` 指向指定 pin commit，且外层 `git status` 不把上游源码当作本仓库文件

#### Scenario: 更新上游版本

- **WHEN** 需要升级 `wekws`
- **THEN** 变更只通过显式 submodule checkout 或 pull 完成，并记录新的 pin commit

### Requirement: 转换前上游工作树必须干净

在转换为 submodule 之前，项目 SHALL 恢复 `wekws` 中已有的本地修改、确认未跟踪下载物已迁出，并保留可追溯的 diff 备份。

#### Scenario: 恢复两处本地修改

- **WHEN** 转换前检查 `wekws` 工作树
- **THEN** `examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml` 与 `examples/hi_xiaowen/s0/run_ctc.sh` 已恢复到上游版本，本地差异已备份到 `docs/plans/backups/wekws_local_changes.diff`

#### Scenario: 未跟踪下载物仍在

- **WHEN** `wekws/examples/hi_xiaowen/s0/mobvoi_kws_transcription/` 尚未迁出
- **THEN** submodule 转换不得执行，校验流程报告上游目录不干净

### Requirement: 下载资源由外层脚本管理

`mobvoi_kws_transcription` SHALL 作为完整下载仓库位于 `work/downloads/mobvoi_kws_transcription/`，并由外层下载脚本创建；项目 MUST 不把它拆分或复制回 `wekws/`。

#### Scenario: 首次下载

- **WHEN** 下载脚本发现目标目录不存在
- **THEN** 脚本将完整仓库下载到 `work/downloads/mobvoi_kws_transcription/`，保留其 `.git`

#### Scenario: 重复执行下载

- **WHEN** 下载脚本发现目标目录已存在
- **THEN** 脚本跳过下载且不修改已有资源

#### Scenario: 训练引用 base checkpoint

- **WHEN** 训练入口解析默认 base checkpoint
- **THEN** 路径为 `work/downloads/mobvoi_kws_transcription/23.pt`，不指向 `wekws/` 内的副本

### Requirement: 上游目录的只读性可验证

项目 SHALL 提供可执行的只读校验，发现 `wekws/` 中存在项目落地文件或本地修改时 MUST 失败。

#### Scenario: 注入 bytecode 缓存

- **WHEN** `wekws/` 内出现 `__pycache__`
- **THEN** 只读校验失败并指出违规路径

#### Scenario: 干净的上游目录

- **WHEN** `wekws/` 仅包含上游源码且 `git -C wekws status --short` 为空
- **THEN** 只读校验通过
