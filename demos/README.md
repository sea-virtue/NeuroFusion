# 独立任务 demo

每个任务使用一个 `demos/<任务名>/` 目录，配置、入口、说明、模型导出和日志都归该任务管理。

| 任务 | 功能 | 入口 |
| --- | --- | --- |
| [slip-recatch](slip-recatch/README.md) | 松爪探索、触觉滑动检测、即时回夹 | `bash demos/slip-recatch/run.sh` |
| [balloon-grip](balloon-grip/README.md) | 悬挂砝码振荡时的双触觉夹持调节 | `bash demos/balloon-grip/run.sh` |

后续 demo 沿用下面的约定：

```text
demos/<任务名>/
  README.md                 操作与参数说明
  run.sh                    单一启动入口
  configs/
    config.example.json     可提交的默认模板
    config.local.json       本机设置，不提交
  models/                   该任务的导出模型，不提交
  outputs/                  该任务的运行记录，不提交

src/neurofusion/demos/<任务名的Python形式>/   实际运行代码
tests/demos/<任务名的Python形式>/            对应软件测试
```

Python 包名使用下划线，例如 `slip_recatch`。夹爪、相机驱动仍在共享的 `src/neurofusion/adapters/`，
训练网络仍在 `src/neurofusion/models/`。通用硬件维护工具放 `tools/<设备>/`；不复制驱动或环境到每个 demo。
顶层 `configs/` 用于共享训练/硬件配置，顶层 `scripts/` 保留通用采集、训练与手动设备工具。

`config.local.json`、模型导出和输出目录均由 `.gitignore` 排除。新 demo 的示例配置应可独立说明单位，
执行时再读取设备量程并换算内部参数。
