import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
HF_HOME = CACHE_DIR / "hf"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
if os.getenv("USE_HF_MIRROR", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import streamlit as st
import streamlit.components.v1 as components
import spacy
from spacy.cli import download as spacy_download
import requests
import re
from fastcoref import FCoref
import pandas as pd

# 页面配置
st.set_page_config(page_title="高级 NLP 架构展示 - 话语分析与指代消解", layout="wide", initial_sidebar_state="expanded")

# --- 核心模型加载逻辑 (单例缓存) ---
@st.cache_resource
def load_spacy():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)

@st.cache_resource
def load_fcoref(model_path):
    """
    支持本地路径优先加载，并增加异常鲁棒性。
    """
    # 强制尝试相对路径，这是在 Windows 上最稳健的加载方式，避免 C: 的冒号冲突
    target = "./f-coref"
    
    # 只有当本地目录确实存在且包含权重文件时才尝试本地加载
    if os.path.exists(target) and any(os.path.exists(os.path.join(target, f)) for f in ['pytorch_model.bin', 'model.safetensors']):
        try:
            # 优先使用带 ./ 的相对路径，绕过 Windows 绝对路径解析 bug
            return FCoref(model_name_or_path=target, device='cpu')
        except Exception as e:
            # 如果相对路径也失败，回退到绝对路径尝试
            try:
                abs_target = os.path.abspath(target).replace("\\", "/")
                return FCoref(model_name_or_path=abs_target, device='cpu')
            except:
                pass
            st.warning(f"本地模型加载失败，将尝试从网络加载或使用在线缓存。详情: {e}")
    
    # 终极回退：使用官方 repo ID
    return FCoref(model_name_or_path='biu-nlp/f-coref', device='cpu')

nlp = load_spacy()

# --- 辅助功能: 高清全页截图 ---
def add_screenshot_feature():
    screenshot_html = """
    <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
    <style>
        .capture-btn {
            background-color: #ff4b4b; color: white; border: none; padding: 0.6rem 1.2rem;
            border-radius: 0.5rem; cursor: pointer; width: 100%; font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;
        }
        .capture-btn:hover { background-color: #ff3333; transform: translateY(-2px); }
    </style>
    <button class="capture-btn" onclick="takeScreenshot()">📸 导出全页分析报告</button>
    <script>
    async function takeScreenshot() {
        const btn = document.querySelector('.capture-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '⏳ 正在渲染长图...';
        btn.disabled = true;

        try {
            const rootDoc = window.parent.document;
            const mainContent = rootDoc.querySelector('[data-testid="stApp"]') || rootDoc.body;
            
            const canvas = await html2canvas(mainContent, {
                useCORS: true, scale: 1.5, backgroundColor: "#ffffff",
                height: mainContent.scrollHeight,
                windowHeight: mainContent.scrollHeight,
                scrollX: 0, scrollY: 0, x: 0, y: 0
            });

            const link = document.createElement('a');
            link.download = `NLP_Report_${new Date().getTime()}.png`;
            link.href = canvas.toDataURL("image/png");
            link.click();
            btn.innerHTML = '✅ 导出成功';
        } catch (err) {
            alert('截图失败: ' + err.message);
            btn.innerHTML = '❌ 重试';
        } finally {
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 3000);
        }
    }
    </script>
    """
    st.sidebar.markdown("### 🛠️ 页面工具")
    with st.sidebar:
        components.html(screenshot_html, height=70)

# --- 模块 1: 话语分割 (EDU) ---
def module_edu_segmentation():
    st.header("模块 1: 话语分割 (EDU 切分)")
    url = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_0601.out.edus"
    
    if 'edu_data' not in st.session_state:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                st.session_state.edu_data = [l.strip() for l in res.text.split('\n') if l.strip()]
        except: st.error("无法获取在线样本数据")

    if 'edu_data' in st.session_state:
        full_text = " ".join(st.session_state.edu_data)
        doc = nlp(full_text)
        
        # 规则切分逻辑
        boundaries = []
        for i, t in enumerate(doc):
            if t.text in [".", "!", "?", ";", ":"] or (t.text == "," and i < len(doc)-1):
                boundaries.append(i)
            elif t.pos_ == "SCONJ" or t.dep_ in ["advcl", "ccomp"]:
                if i > 0: boundaries.append(i-1)
        
        boundaries = sorted(list(set(boundaries)))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("规则基线结果")
            start = 0
            tokens = [t.text for t in doc]
            for b in boundaries:
                seg = " ".join(tokens[start:b+1])
                st.markdown(f'<div style="border:1px solid #ddd; padding:8px; margin-bottom:4px; border-radius:4px; background:#f8f9fa;">{seg} <span style="color:red; font-weight:bold;">[B]</span></div>', unsafe_allow_html=True)
                start = b + 1
        with col2:
            st.subheader("NeuralEDUSeg 真实标注")
            for edu in st.session_state.edu_data:
                st.markdown(f'<div style="border:1px solid #ddd; padding:8px; margin-bottom:4px; border-radius:4px; background:#e9ecef;">{edu} <span style="color:green; font-weight:bold;">[GT]</span></div>', unsafe_allow_html=True)

# --- 模块 2: 篇章分析 ---
def module_discourse_analysis():
    st.header("模块 2: 浅层篇章分析与论据提取")
    default = "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs and new products - although weaker foreign currencies reduced the company's earnings."
    text = st.text_input("分析句子：", value=default)
    
    conn_map = {"although": "Comparison", "but": "Comparison", "because": "Contingency", "since": "Contingency", "when": "Temporal", "and": "Expansion"}
    
    found = None
    for c, t in conn_map.items():
        m = re.search(rf'\b{c}\b', text, re.I)
        if m:
            found = (m.group(), t, m.start(), m.end())
            break
            
    if found:
        st.success(f"识别到连接词: **{found[0]}** [{found[1]}]")
        arg1 = re.sub(r'[\s\-\,]+$', '', text[:found[2]])
        arg2 = text[found[3]:].strip()
        
        c1, c2 = st.columns(2)
        c1.info(f"**Arg1 (前置论据):**\n\n{arg1}")
        c2.warning(f"**Arg2 (后置论据):**\n\n{arg2}")
        
        high = text[:found[2]] + f'<span style="background:yellow; padding:2px; font-weight:bold; color:black;">{found[0]} [{found[1]}]</span>' + text[found[3]:]
        st.markdown(f"**可视化标注:**\n\n{high}", unsafe_allow_html=True)

# --- 模块 3: 指代消解 ---
def module_coreference_resolution():
    st.header("模块 3: 指代消解可视化")
    
    # 本地模型路径优先检测
    st.sidebar.markdown("### 🤖 模型设置")
    local_path = st.sidebar.text_input("模型路径", value="./f-coref")
    
    text = st.text_area("分析文本：", value="Barack Obama was born in Hawaii. He was the 44th president. His wife is Michelle. They have two daughters. The daughters love their father.", height=120)
    
    if st.button("开始端到端分析"):
        model = None
        use_mock = st.session_state.get('use_mock', False)
        
        if not use_mock:
            try:
                with st.spinner("正在加载模型并分析..."):
                    model = load_fcoref(local_path)
            except Exception as e:
                st.error(f"""
                **模型加载失败：{str(e)}**
                
                这通常是因为 `f-coref` 目录下缺少 `pytorch_model.bin` (约 400MB) 
                或 Git LFS 指针文件不完整。
                """)
                if st.button("切换离线演示模式"):
                    st.session_state.use_mock = True
                    st.rerun()
                return

        with st.spinner("正在进行端到端分析..."):
            if use_mock:
                # 预设数据展示
                clusters = [
                    [(0, 12), (32, 34), (43, 45), (105, 111)], # Obama/He/He/father
                    [(76, 90), (100, 109)] # Michelle/daughters
                ]
            else:
                preds = model.predict(texts=[text])
                # fastcoref 的 get_clusters 默认返回字符串，设置 as_strings=False 获取索引
                clusters = preds[0].get_clusters(as_strings=False)

                colors = ["#ff7675", "#74b9ff", "#55efc4", "#ffeaa7", "#a29bfe", "#fab1a0"]
                mentions = []
                for i, cluster in enumerate(clusters):
                    for s, e in cluster:
                        mentions.append((s, e, i))
                mentions.sort()

                # HTML 渲染
                html, last = "", 0
                for s, e, idx in mentions:
                    if s < last: continue
                    html += text[last:s] + f'<span style="background:{colors[idx % len(colors)]}; padding:2px 4px; border-radius:3px; font-weight:500;">{text[s:e]}</span>'
                    last = e
                html += text[last:]
                
                st.subheader("高亮结果")
                st.markdown(f'<div style="line-height:2.2; font-size:1.1rem; border:1px solid #eee; padding:1.5rem; border-radius:10px;">{html}</div>', unsafe_allow_html=True)
                
                st.subheader("实体等价类")
                for i, cluster in enumerate(clusters):
                    m_list = [text[s:e] for s, e in cluster]
                    st.write(f"**Cluster {i+1}**: {m_list}")

def main():
    st.title("高级 NLP 综合架构展示")
    add_screenshot_feature()
    
    tabs = st.tabs(["话语分割 (EDU)", "篇章关系提取", "指代消解分析"])
    with tabs[0]: module_edu_segmentation()
    with tabs[1]: module_discourse_analysis()
    with tabs[2]: module_coreference_resolution()

if __name__ == "__main__":
    main()
