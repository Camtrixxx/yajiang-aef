# Yajiang AEF 在线演示系统

## 1. 启动方式

进入容器：

```bash
docker exec -it heyuhang-dl bash
```

启动演示服务：

```bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_demo_server.sh
```

默认配置：

```text
port: 7860
device: auto
ASCEND_RT_VISIBLE_DEVICES: 4
```

也就是默认使用后四张卡中的物理 `4` 号卡。

如果要指定端口：

```bash
PORT=7861 bash scripts/run_demo_server.sh
```

如果要显式换卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=5 PORT=7860 bash scripts/run_demo_server.sh
```

## 2. 当前后台服务

如果使用 Codex 已经启动的后台服务，可以直接访问：

```text
http://localhost:7860
```

如果 VS Code 提示端口转发，选择转发 `7860`。

## 3. 页面功能

页面左侧可以选择：

- patch index；
- 常用样本；
- 随机样本。

页面右侧展示：

- S2 RGB 输入；
- AEF embedding PCA；
- DEM target / reconstruction / error；
- WorldCover target / reconstruction；
- JRC Water target + predicted contour；
- 单样本指标；
- WorldCover 类别统计；
- few-shot 曲线；
- embedding PCA；
- 混淆矩阵。

## 4. 停止服务

```bash
pkill -f scripts/serve_demo.py
```

## 5. 说明

这个系统用于演示和诊断，不是独立测试集评估。当前使用的是 v0.3c deploy 模型：

```text
outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt
```

后续如果有新的模型版本，可以在启动时修改 `--deploy-model`，或者新建对应的 run 脚本。
