# 2027 评测证据索引

此目录在GitHub中仅发布本说明；下表所列原始JSON/NPZ按用户要求保留在desktop，未上传。公开清单记录文件名、大小与SHA-256。

这些是实际产生的历史 JSON/NPZ。它们涵盖探索、修复前口径、修复后口径和诊断参数，不构成一个可以直接合并的统计样本。

## 优先阅读

| 文件 | 含义 |
|---|---|
| `slalom_pursuit_v7_smoke20.json` | 059950+路径跟踪，绕桩19/20，仍需判据复核和大样本验收 |
| `beam_pursuit_v3_smoke20.json` | 规则单木桥14/20，6次桥外落足失败 |
| `low_frame_064242_x200_stage1_nominal_stage192.json` | 宽松Stage 1 Isaac nominal 192/192，不代表290mm比赛框架 |
| `low_frame_stage1_hard_ft2_trainmatched_4x48_seeds20261127_30.json` | 同批Isaac 064341/064400/064500/064540随机初态对照 |
| `low_frame_064341_stage1_mujoco_corrected_default_unclipped_smoke24_seed2027.json` | 修正判据、原设阻尼、无额外目标裁剪，2/24 |
| `low_frame_064341_stage1_mujoco_corrected_damping3_unclipped_regression120_seed2028.json` | 诊断阻尼3×下120/120，不能用于正式晋级 |
| `low_frame_064400_stage1_mujoco_corrected_damping3_unclipped_regression120_seed2028.json` | 备选策略，同一诊断条件120/120 |
| `low_frame_064540_stage1_mujoco_corrected_damping4_unclipped_smoke24_seed2027.json` | 续训末点退化反例，0/24 |
| `low_frame_stage1_dynamics_replay_audit.json` | 同目标开环、初态/历史对齐、接触参数扫描与附件校验 |

## 使用限制

- `corrected`结果使用更严格的门内通过与5ms子步接触采样；旧结果不能直接比较。
- `damping2/3/4`是物理敏感性扫描，未经实机标定；`classification`字段的自动建议不等于研究验收结论。
- `nominal`与`trainmatched`初态分布不同；不同run中的相同编号不能视为同一权重。
- 某些探索JSON没有记录完整argv/物理开关；文件名只能提供线索，复现时以配套状态文档和轨迹为准。部分临时诊断脚本未归档，本清单不承诺所有历史实验可一键重跑。
- 原设阻尼本身也不是已验证的真实硬件参数。250N限高冲击阈值为临时代理；框架位移与安全限值待实测。
- JSON/NPZ的SHA-256用于检查文件一致性，不代表结果已通过实机验收。

在持有原始数据的desktop上，从仓库根运行 `python scripts/audit_robocon_2027_artifacts.py --verify` 检查本地证据。只clone公开仓库时，此命令会如实报告缺少原始数据。新的实验应另存，后续收尾时显式更新清单。

状态与计划：

- [执行状态](../../docs/2027_ROBOCON_EXECUTION_STATUS.md)
- [下一步工作包与验收](../../docs/2027_ROBOCON_NEXT_STEPS_20261015.md)
- [权重及证据清单](../../docs/2027_ROBOCON_ARTIFACT_MANIFEST.json)
