# Colab Pro 长训 Playbook（InnerLogic-Surg）

> 2026-11-22 调查验证后定稿。背景：当晚 Colab 第三轮续训连丢两次会话、重建约 2.5 小时、烧 6.68 计算单元、零存档；ep11 曾跑到验证 Dice 0.969 但检查点随会话丢失。Modal 续训已接手，Colab 下次长训按此执行。

## 1. 正确架构（一句话）

**Drive 是持久 volume，会话只是算力。** 数据集、代码、检查点、结果全部住在 Drive；会话 `/content` 只放解压后的工作副本和当轮临时文件。会话死，只丢正在跑的那一轮。

```
MyDrive/ILS-Colab/                <- 建议的 Drive 目录（本次未创建，下次建）
  data/sea.zip                    <- 9,877,132,195 字节，已在 Drive S01 文件夹
  code/                           <- 训练脚本 + 划分 manifests
  checkpoints/best_ep{NN}.pt      <- 每轮最佳，训完即写
  results/
会话 /content/
  drive/                          <- 挂载点（每次新会话挂一次）
  ils/sea.zip  <- 从 Drive 拷过来解压
  ils/checkpoints/                <- 当轮检查点，先写本地再同步到 Drive
```

## 2. 主流用法（已核实）

- 每个新会话第一件事就是挂载 Drive：`drive.mount('/content/drive')`，再把项目文件夹拷到本地盘跑。
- 检查点每轮最佳都写 Drive，不是跑完才写。断在第 47 轮、没有检查点，是每个 Colab 用户都交过的学费。
- 数据集放 Drive 一份 zip，`unzip` 到 `/content` 用；不要每次重下 9.8 GB。
- 会话上限与回收是平台行为（Pro 比免费长，但一样会回收），不要试图“保活”——设计成可断点续跑才是正解。

## 3. 我们挂载失败的根因（已查实）

`drive.mount` 需要在**浏览器里**完成一次 OAuth 同意（完整 `drive` 权限）。在浏览器 UI 里点一下 Allow，几秒就挂上；**在 CLI/无头环境里没有这个弹窗**，只能等 120 秒超时然后报 `ValueError: mount failed`。同款故障见 googlecolab/colab-vscode#264（VS Code 里挂载同样 120 秒超时，浏览器里即时成功）。

- 6 次失败全是这个机制：CLI 拦截到授权请求、打印出链接，等用户在聊天另一头看到链接、打开浏览器、点同意，再等 CLI 收到回车——三步要在 120 秒内走完。11 月 7 日那次用户点了、看到了成功页，还是失败：回车确认那步没赶上。
- 已排除的嫌疑：My Drive 根目录只有 **37 个**文件（FAQ 说根目录 >1 万项才会挂载超时，不适用）。
- 已排除的捷径：**CLI 自身的 OAuth token 不能复用于 Drive**。实测 `~/.config/colab-cli/token.json` 的 scope 只有 `openid / userinfo / cloud-platform / colaboratory / drive.file`，调 Drive API 直接 403（无 quota project / 权限不足）。挂载走的是另一套带完整 `drive` scope 的 OAuth client，必须用户在浏览器里重新授权一次，不存在“免点击复用”。
- 补充：`drive.mount` 默认是 ephemeral 凭证，跟浏览器会话绑定；CLI 会话本来就没有浏览器挂着，就算某次蒙上了也不可靠。

## 4. 可靠的挂载路径（按推荐度排序）

### A. 浏览器内挂载（首选，主流用法）

1. 建会话：`~/.local/bin/colab new -s <名字> --gpu L4`（CPU 验证用不加 `--gpu`）
2. 取浏览器链接：`~/.local/bin/colab url -s <名字>`（已验证：能拿到形如 `https://colab.research.google.com/notebooks/empty.ipynb?dbu=...#datalabBackendUrl=...` 的链接，指向同一台 VM）
3. **用户在浏览器里打开该链接**，在 notebook 里跑一个 cell：`from google.colab import drive; drive.mount('/content/drive')`
4. 同一浏览器里点 Allow（Google 账号已登录，通常一次点击，不用重新输密码）
5. 挂上后 `ls /content/drive/MyDrive/` 能看到文件即成

- 谁操作：第 3–4 步用户在浏览器里点一次；其余 agent 做。
- 授权有效期：挂载跟当前 runtime 走，会话死要重挂；但 Google 账号里对 Colab 应用的授权长期有效，下次一般仍是一次点击。
- 费用：挂载本身 0 费用。

### B. CLI drivemount（备用，脆弱）

`~/.local/bin/colab drivemount -s <名字>` 会打印授权链接并等回车。用户必须在 **120 秒内**点完链接、agent 立即在 PTY 里发回车。6 次失败证明这条路经不起聊天中转，只在用户守在屏幕前时用。

### C. rclone（一次授权、之后免点击）

`rclone config` 建一个 Drive remote，授权时用户在浏览器开链接、把 code 粘贴回来一次；refresh token 存 `~/.config/rclone/rclone.conf`（600 权限），之后 `rclone mount` 或 `rclone copy/sync` 全程免点击。适合“不想每轮都点一次”的场景。注意：code 是凭证，粘贴走安全输入，不走聊天正文；本次未做，需要时再走这一步。

### D. 不推荐

- 复用 CLI token：已证伪（403）。
- 服务账号：手头没有，不引入新账号体系。
- 反复 `colab update` 想修好挂载：CLI 0.7.4 带 411 补丁，**绝不 update**。

## 5. 会话丢失后的恢复步骤（目标：分钟级）

1. 新建会话：`colab new -s <名字> --gpu L4`
2. 按 §4-A 挂载 Drive（用户点一次，约 1 分钟）
3. 从 Drive 拷 zip 到本地并解压：`cp /content/drive/MyDrive/ILS-Colab/data/sea.zip /content/ils/ && unzip -q ...`（本地解压约几分钟；不要重下）
4. 从 Drive 拷回最佳检查点：`cp /content/drive/MyDrive/ILS-Colab/checkpoints/best_ep*.pt /content/ils/`
5. 从该 epoch 续训。损失 = 正在跑的那一轮。

## 6. 检查点保存规范

- 每轮验证完、只要是新最佳：先写本地 `/content/ils/checkpoints/best_ep{NN}.pt`，**立即** `cp` 到 Drive `checkpoints/`；再更新进度文件（写明 Drive 路径与 epoch）。
- 训练脚本的 `--resume` 必须默认先去 Drive 找最佳检查点，找不到才报错停下，不许静默从头训。
- 余额每轮记一笔；单轮预计 L4 约 14 分钟 ≈ 0.45 计算单元。

## 7. 旧用法错因对照

| 旧用法（11-22 事故） | 正确用法 |
|---|---|
| 数据集每次重下 9.8 GB + 解压 | zip 常驻 Drive，会话内只解压 |
| 检查点 28 块 20 MB 分块上传进会话 | 检查点常驻 Drive，会话内只读 |
| 所有东西只放 `/content` | `/content` 只放工作副本 |
| 挂载失败 5 次后绕开 Drive | 挂载是必经步骤，失败要修流程（浏览器内挂载），不是绕开 |
| 检查点跑完才取回 | 每轮最佳立即写 Drive |
| 会话丢 = 2.5 小时重建 + 6.68 单元 | 会话丢 = 挂载 + 拷文件，分钟级 |

## 8. 本次验证状态（诚实记录）

- 已验证：CPU 会话创建（`ils-playbook-verify`，Standard CPU）、`colab url` 能拿到浏览器接入链接、会话内核可执行代码、Drive 未挂载时 `/content` 只有 `.config` 与 `sample_data`、停会话后余额纹丝不动（192.29，约 5 分钟 CPU 会话花费 0.00 计算单元，限额 0.5 未动）。
- **未完成：marker 跨会话存活证明。** 原因：三条挂载路径里，A 与 B 都要用户在浏览器里点一次授权，C 也要一次 code 粘贴；本次无用户点击可用，未盲试。待用户点一次后按下面流程补齐：
  1. `colab new -s ils-playbook-verify2`（CPU，不加 `--gpu`）
  2. `colab url -s ils-playbook-verify2`，用户在浏览器打开，cell 里 `drive.mount('/content/drive')` 并点 Allow
  3. 会话内写 marker：`echo <时间戳> > /content/drive/MyDrive/ils-playbook-marker.txt`，`ls` 确认
  4. `colab stop -s ils-playbook-verify2`，再 `colab new -s ils-playbook-verify3`
  5. 重复步骤 2 挂载，`cat /content/drive/MyDrive/ils-playbook-marker.txt`，时间戳一致即证明持久性
  6. `rm /content/drive/MyDrive/ils-playbook-marker.txt`，删 marker，收尾
- 本次未在 Drive 里创建、修改、删除任何内容，无清理欠账。

## 9. 下次长训前必须改的三个点

1. **先挂载、再训练**：会话起来第一件事是浏览器内挂载 Drive，挂不上不开训。
2. **数据集与检查点常驻 Drive**：zip 与每轮最佳检查点只以 Drive 为准，会话只作计算。
3. **断点续跑写进脚本**：训练入口默认从 Drive 读最佳检查点续跑；进度文件每轮写 Drive 路径与余额，丢会话后恢复不超过 10 分钟。

## 10. 2026-11-23 浏览器挂载验证（补充）

- **绑定本身通了**：云端浏览器打开 `colab url` 链接，在笔记本里跑 `drive.mount('/content/drive')`，经两次手机验证（Google 先后要求点数字 54、29）+ OAuth 同意后，cell 输出 `Mounted at /content/drive`；换一个浏览器任务重进同一链接，`/content/drive` 仍在，挂载稳定。
- **新发现：CLI 会话 ≠ 浏览器笔记本运行时**：`!hostname` 对照——浏览器侧 `0d1a47364fe4`，CLI `colab exec` 侧 `4328f590d847`，两台不同 VM。`colab url` 给的链接（带 dbu + datalabBackendUrl）没有把云端浏览器带到 CLI 建的那台 VM 上，playbook §4-A“url 指向同一台 VM”的假设在云端浏览器下不成立。
- **含义**：挂载与训练必须在同一运行时里；CLI exec 看不到浏览器侧的挂载，反之亦然。marker 跨会话存活证明改走浏览器侧：浏览器 cell 写 marker → 断开并删除浏览器运行时 → 重开链接、二次挂载 → 浏览器 cell 读 marker。
- **验证方法**：任何“同一台”的断言都以 `!hostname` 对照为准，不要凭链接参数推断。
- **费用**：本次 CPU 会话 + exec 共耗 0.03 计算单元（192.29 → 192.26），约等于零。
- **持久性证明已完成（2026-11-23 23:05）**：旧运行时 0d1a47364fe4 断开删除后，新运行时 55b963c02ae5 二次挂载，读回 marker 时间戳 2026-11-23T14:59:49Z 逐字一致（marker 内记的 runtime 与新 hostname 不同，确为旧运行时所写），随后 marker 已删。结论：**Drive 当持久 volume 的路子走通了**，会话丢了文件不丢。
- **二次挂载不用手机验证**：第一次挂载 Google 要求手机点 Yes + 数字（54、29 两次）；新运行时二次挂载只点了两次 Continue，全程无手机验证——应用授权一次之后就省事了。
- **环境不存 Drive（2026-11-23 确认）**：标准做法是数据、笔记本、代码、requirements.txt、检查点放 Drive，环境本身不放。Colab 基础镜像自带 torch 等常见库，新会话 pip 重装一般只要几分钟；11-22 那晚重建烧钱的根因是 9.8GB 数据集每次重下 + 检查点分块重传，不是环境重建。只在依赖需要编译（flash-attn、自定义 CUDA 扩展）或大体积冷门 wheel 时，才值得把 wheel 文件缓存在 Drive 里离线装；直接 `pip install --target` 装到 Drive 里一般不推荐，Drive 是 FUSE，几千个小文件 import 很慢。
