import json
import gradio as gr
import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from vision import VisionModule  # <--- Import mới

from agent import ToolUseAgent
from tools import TOOLS_SCHEMA

MODEL_OPTIONS = {
    "Qwen Agent": "/home/manh/Projects/temp/CS431/src_agent/agent_model_weights/checkpoint-318",
    "Vietnamse Qwen 2.5 Math (1.5B)": "piikerpham/Vietnamese-Qwen2.5-math-1.5B", 
    "Qwen 2.5 Math (1.5B)": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen 2.5 Math (7B)": "Qwen/Qwen2.5-Math-7B-Instruct",
}

current_model = None
current_tokenizer = None
current_agent = None
loaded_model_name = ""
vision_module = VisionModule()

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

def solve_math_problem(model_select, question, image_path, show_reasoning, temperature, max_tokens):
    global current_agent, loaded_model_name, vision_module
    
    reasoning_display = ""
    full_question = question
    
    if image_path is not None:
        if current_model is not None:
            print("⚠️ Tạm thời unload Math Model để chạy Vision...")
            clean_memory()
            
        status_msg = "👁️ Đang đọc ảnh với Vintern-1B..."
        print(status_msg)
        reasoning_display += f"### 👁️ Xử lý Hình ảnh (Vintern-1B)\n"
        
        try:
            extracted_text = vision_module.extract_text_from_image(image_path)
            reasoning_display += f"> **Nội dung trích xuất:**\n{extracted_text}\n\n---\n"
            
            full_question = f"{extracted_text}\n\n{question}"
        except Exception as e:
            reasoning_display += f"> ❌ Lỗi đọc ảnh: {str(e)}\n\n---\n"
    
    needs_reload = False
    if loaded_model_name != model_select: needs_reload = True
    try:
        current_model.device 
    except:
        needs_reload = True
        
    if needs_reload:
        status = load_model_pipeline(model_select)
        if "Lỗi" in status: return status, reasoning_display

    if not current_agent: return "Lỗi: Không thể khởi tạo Agent.", reasoning_display
    if not full_question.strip(): return "Vui lòng nhập câu hỏi hoặc upload ảnh.", reasoning_display

    current_agent.generation_cfg = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
    }

    try:
        print(f"🤖 Agent đang suy luận với model: {loaded_model_name}")
        conversations, final_answer = current_agent.inference(full_question)
        
        if show_reasoning:
            step_count = 1
            for msg in conversations:
                role = msg['role']
                content = str(msg['content'])
                
                if role == 'assistant':
                    if "<tool_call>" in content:
                        parts = content.split("<tool_call>")
                        thought = parts[0].strip()
                        tool_code = parts[1].replace("</tool_call>", "").strip()
                        
                        reasoning_display += f"### 🧠 Bước {step_count}: Suy luận\n"
                        if thought: reasoning_display += f"{thought}\n\n"
                        reasoning_display += f"**⚡ Hành động:**\n```json\n{tool_code}\n```\n\n"
                        step_count += 1
                    else:
                        if content.strip() != final_answer.strip():
                            reasoning_display += f"### 🧠 Bước {step_count}: Suy luận\n{content}\n\n"
                            step_count += 1

                elif role == 'tool':
                    clean_res = content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                    reasoning_display += f"### 🔧 Kết quả Công cụ\n> {clean_res}\n\n---\n"

        if not final_answer:
            final_answer = conversations[-1]['content']

        return final_answer, reasoning_display

    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}", reasoning_display

# --- GRADIO UI ---
css = """
#reasoning_box { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; max-height: 500px; overflow-y: auto; }
#status_box { font-weight: bold; color: #2e7d32; }
"""

with gr.Blocks(title="Math Agent + Vintern Vision", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🧮 Hệ thống Giải Toán Đa Phương Thức (Vintern + Qwen)")
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Cấu hình Model")
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="My Finetune (Checkpoint 318)", 
                    label="Math Agent Model",
                    interactive=True
                )
                load_status = gr.Textbox(label="Trạng thái", value="Sẵn sàng...", elem_id="status_box", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 2. Nhập Đề Bài")
                # Dùng type="filepath" để tương thích với hàm load_image của Vintern
                image_input = gr.Image(type="filepath", label="📸 Upload ảnh bài toán")
                question_input = gr.Textbox(lines=3, placeholder="Nhập thêm yêu cầu (VD: Giải chi tiết bài toán trên)...", label="Câu hỏi bổ sung")
            
            with gr.Accordion("⚙️ Cấu hình nâng cao", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.5, label="Temperature")
                max_tokens = gr.Slider(128, 2048, 1024, label="Max Tokens")
                show_reasoning = gr.Checkbox(True, label="Hiện suy luận")
            
            solve_btn = gr.Button("🚀 GIẢI BÀI NGAY", variant="primary", size="lg")

        with gr.Column(scale=5):
            gr.Markdown("### 🏁 Kết quả cuối cùng")
            answer_output = gr.Textbox(label="", interactive=False, lines=3)
            gr.Markdown("### 🧠 Quá trình suy luận (Vision -> Thought -> Tools)")
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
    
    demo.load(fn=load_model_pipeline, inputs=[model_selector], outputs=[load_status])

if __name__ == "__main__":
    demo.launch(share=True)