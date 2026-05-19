# 高级 NLP 架构展示：话语分析与指代消解（Streamlit）

包含三个模块：话语分割（EDU）、浅层篇章分析、指代消解可视化（Coreference Resolution）。

## 在线使用（Streamlit Community Cloud）

1. 将本目录作为一个独立仓库推到 GitHub（仓库根目录包含 `app.py`）。
2. 打开 https://streamlit.io/cloud ，选择该仓库并部署。
3. 部署成功后会得到一个公开链接，其他人无需你本机开机即可直接访问。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 模型下载与缓存

- spaCy 模型 `en_core_web_sm`：首次运行自动下载。
- 指代消解模型：默认优先尝试本地 `./f-coref`（若存在完整权重）；否则会从 Hugging Face 自动拉取 `biu-nlp/f-coref` 并缓存到本项目 `.cache/hf/`。

## 可选：开启镜像

如果你所在网络访问 Hugging Face 较慢，可以在部署平台配置环境变量：

- `USE_HF_MIRROR=1`：启用 `https://hf-mirror.com` 作为下载镜像。
