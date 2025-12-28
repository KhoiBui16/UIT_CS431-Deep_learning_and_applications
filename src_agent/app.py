import json
import gradio as gr
import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from agent import ToolUseAgent
from tools import TOOLS_SCHEMA

MODEL_OPTIONS = {
    "Qwen Agent": "/home/manh/Projects/temp/CS431/src_agent/agent_model_weights/checkpoint-318",
    "Qwen 2.5 Math (1.5B)": "piikerpham/Vietnamese-Qwen2.5-math-1.5B", 
    "Qwen 2.5 Math (7B)": "Qwen/Qwen2.5-Math-7B-Instruct",
}

current_model = None
current_tokenizer = None
current_agent = None
loaded_model_name = ""

def clean_memory():
    """Hàm dọn dẹp VRAM triệt để"""
    global current_model, current_tokenizer, current_agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def load_model_pipeline(model_key):
    global current_model, current_tokenizer, current_agent, loaded_model_name
    
    if loaded_model_name == model_key and current_agent is not None:
        return f"✅ Model '{model_key}' đã sẵn sàng!"

    print(f"🔄 Đang chuyển đổi sang model: {model_key}...")
    
    if current_model is not None:
        del current_model
        del current_tokenizer
        del current_agent
        clean_memory()

    model_path = MODEL_OPTIONS[model_key]
    try:
        print(f"⏳ Đang load từ: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if "7B" in model_key or "7b" in model_path.lower():
            print("⚠️ Phát hiện Model lớn (7B). Đang bật chế độ 4-bit Quantization để tiết kiệm VRAM...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config, 
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        current_model = model
        current_tokenizer = tokenizer
        current_agent = ToolUseAgent(model, tokenizer, tools_metadata=TOOLS_SCHEMA)
        loaded_model_name = model_key
        
        print(f"✅ Load thành công: {model_key}")
        return f"✅ Đã chuyển sang: {model_key}"
        
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return f"❌ Lỗi: {str(e)}"

def solve_math_problem(model_select, question, image, show_reasoning, temperature, max_tokens):
    global current_agent, loaded_model_name
    
    if loaded_model_name != model_select or current_agent is None:
        status = load_model_pipeline(model_select)
        if "Lỗi" in status: return status, ""

    if not current_agent: return "Lỗi: Không thể khởi tạo Agent.", ""

    context_image = ""
    if image is not None:
        context_image = "\n(Người dùng đính kèm ảnh, nhưng chưa có module OCR)."
    
    full_prompt = question + context_image
    if not full_prompt.strip(): return "Vui lòng nhập câu hỏi.", ""

    current_agent.generation_cfg = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
    }

    try:
        print(f"🤖 Agent đang suy luận với model: {loaded_model_name}")
        
        conversations, final_answer = current_agent.inference(full_prompt)
        
        reasoning_display = ""
        
        if show_reasoning:
            step_count = 1
            for msg in conversations:
                role = msg['role']
                content = str(msg['content'])
                
                if role in ['system', 'user']:
                    continue
                
                if role == 'assistant':
                    if "<tool_call>" in content:
                        parts = content.split("<tool_call>")
                        thought = parts[0].strip()
                        tool_code = parts[1].replace("</tool_call>", "").strip()
                        
                        reasoning_display += f"### 🧠 Bước {step_count}: Suy luận\n"
                        if thought:
                            reasoning_display += f"{thought}\n\n"
                        else:
                            reasoning_display += "(Model quyết định gọi công cụ ngay lập tức...)\n\n"
                            
                        reasoning_display += f"**⚡ Hành động (Gọi Tool):**\n```json\n{tool_code}\n```\n\n"
                        step_count += 1
                    else:
                        if content.strip() == final_answer.strip():
                            pass 
                        else:
                            reasoning_display += f"### 🧠 Bước {step_count}: Suy luận\n{content}\n\n"
                            step_count += 1

                elif role == 'tool':
                    clean_res = content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                    reasoning_display += f"### 🔧 Kết quả Công cụ\n> {clean_res}\n\n---\n"

        if not final_answer:
            final_answer = conversations[-1]['content']

        return final_answer, reasoning_display

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Lỗi hệ thống: {str(e)}", ""

css = """
#reasoning_box { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; max-height: 500px; overflow-y: auto; }
#status_box { font-weight: bold; color: #2e7d32; }
"""

with gr.Blocks(title="Math Agent Multi-Model Demo", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🧮 Hệ thống Giải Toán Thông Minh (Multi-Model Switcher)")
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Chọn Model")
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="My Finetune (Checkpoint 318)", 
                    label="Model AI",
                    interactive=True
                )
                load_status = gr.Textbox(label="Trạng thái System", value="Chưa khởi động...", elem_id="status_box", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 2. Nhập Đề Bài")
                image_input = gr.Image(type="pil", label="Ảnh đề bài (Tùy chọn)")
                question_input = gr.Textbox(lines=4, placeholder="Nhập bài toán...", label="Nội dung")
            
            with gr.Accordion("⚙️ Cấu hình", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.5, label="Temperature")
                max_tokens = gr.Slider(128, 2048, 1024, label="Max Tokens")
                show_reasoning = gr.Checkbox(True, label="Hiện suy luận")
            
            solve_btn = gr.Button("🚀 GIẢI BÀI NGAY", variant="primary", size="lg")

        with gr.Column(scale=5):
            gr.Markdown("### 🏁 Kết quả")
            answer_output = gr.Textbox(label="", interactive=False, lines=3)
            gr.Markdown("### 🧠 Quá trình suy luận")
            reasoning_output = gr.Markdown(elem_id="reasoning_box")

    model_selector.change(
        fn=load_model_pipeline,
        inputs=[model_selector],
        outputs=[load_status]
    )

    solve_btn.click(
        fn=solve_math_problem,
        inputs=[model_selector, question_input, image_input, show_reasoning, temperature, max_tokens],
        outputs=[answer_output, reasoning_output]
    )

    demo.load(
        fn=load_model_pipeline, 
        inputs=[model_selector], 
        outputs=[load_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)