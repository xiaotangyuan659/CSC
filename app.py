"""
中文拼写纠错 - Gradio 前端界面
运行方式：python app.py
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
from sft_test import correct


def correct_interface(source_text: str, error_type: str) -> str:
    """
    Gradio 回调函数，接收用户输入，返回纠错结果。
    """
    if not source_text or not source_text.strip():
        return "请输入待纠错的句子。"

    try:
        result = correct(source_text.strip(), error_type)
        return result if result else "模型未能生成有效输出。"
    except Exception as e:
        return f"出错了：{e}"


# 错误类型选项（与训练数据格式保持一致）
error_type_choices = [
    "形近错字",
    "音近错误",
    "音近字混淆",
    "形近字混淆",
    "混合错误",
]

# 构建 Gradio 界面
with gr.Blocks(
    title="中文拼写纠错",
    css="""
    #title {text-align: center; font-size: 1.6em; font-weight: bold; color: #2c3e50;}
    #subtitle {text-align: center; color: #7f8c8d; margin-bottom: 1.5em;}
    .gr-button-primary {background: #3498db; color: white; font-size: 1.1em;}
    .output-box {font-size: 1.2em; font-weight: 500; color: #27ae60; background: #f0fff4; border: 1px solid #a3d9a5;}
    """
) as demo:
    gr.Markdown("""
    <div id="title">中文拼写纠错</div>
    <div id="subtitle">基于 NanoLlama SFT 模型的中文文本纠错工具</div>
    """, elem_id="header")

    with gr.Row():
        with gr.Column(scale=3):
            source_text = gr.Textbox(
                label="待纠错句子",
                placeholder="请输入包含错误的句子，例如：令天天气很号",
                lines=3,
                info="输入你想要纠正的句子",
            )
        with gr.Column(scale=1):
            error_type = gr.Dropdown(
                label="错误类型",
                choices=error_type_choices,
                value="形近错字",
                info="选择错误的类型（影响 Prompt 构建）",
            )

    submit_btn = gr.Button("开始纠错", variant="primary")

    output = gr.Textbox(
        label="纠错结果",
        lines=2,
        interactive=False,
        elem_classes=["output-box"],
    )

    examples = gr.Examples(
        examples=[
            ["令天天气很号。", "形近错字"],
            ["这是一个晴朗的旱上。", "形近错字"],
            ["我非常喜欢学序网络。", "音近错误"],
        ],
        inputs=[source_text, error_type],
        label="示例",
    )

    submit_btn.click(
        fn=correct_interface,
        inputs=[source_text, error_type],
        outputs=output,
    )

    source_text.submit(
        fn=correct_interface,
        inputs=[source_text, error_type],
        outputs=output,
    )

if __name__ == "__main__":
    print("启动 Gradio 服务，请访问 http://127.0.0.1:7860")
    demo.launch()
