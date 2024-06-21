import gradio as gr
import os
from dotenv import load_dotenv
from utils import generate_with_references, generate_together_stream, inject_references_to_messages
from functools import partial
from rich import print
from time import sleep

# Load environment variables
load_dotenv()

# Ollama-specific environment variables
os.environ['OLLAMA_NUM_PARALLEL'] = '4'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '4'

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")

MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
ROUNDS = int(os.getenv("ROUNDS", 1))
MULTITURN = os.getenv("MULTITURN", "True") == "True"

MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

default_reference_models = [MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]

def moa_generate(prompt, history, aggregate_model, reference_models, temperature, max_tokens, rounds, system_prompt):
    log_output = ""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": "user" if i % 2 == 0 else "assistant", "content": str(msg)} for i, msg in enumerate(history)])
    messages.append({"role": "user", "content": str(prompt)})

    data = {
        "instruction": [messages] * len(reference_models),
        "references": [""] * len(reference_models),
        "model": reference_models,
    }

    for round in range(rounds):
        log_output += f"\nRound {round + 1}:\n"
        
        for i, model in enumerate(reference_models):
            try:
                result = process_fn({"instruction": data["instruction"][i], "model": model}, temperature, max_tokens)
                if result is None or 'output' not in result:
                    raise ValueError(f"Invalid result from {model}: {result}")
                data["references"][i] = result["output"]
                log_output += f"  {model}:\n{result['output']}\n\n"
            except Exception as e:
                log_output += f"  Error with {model}: {str(e)}\n"
                data["references"][i] = f"Error: {str(e)}"

    aggregate_messages = inject_references_to_messages(messages, data["references"])

    try:
        output = generate_with_references(
            model=aggregate_model,
            messages=aggregate_messages,
            references=data["references"],
            temperature=temperature,
            max_tokens=max_tokens,
            generate_fn=generate_together_stream,
            api_base=API_BASE_2,
            api_key=API_KEY_2
        )
        return output, log_output
    except Exception as e:
        log_output += f"Error with aggregate model: {str(e)}\n"
        return f"Error: {str(e)}", log_output

def process_fn(item, temperature, max_tokens):
    try:
        model = item["model"]
        messages = item["instruction"]
        
        output = generate_with_references(
            model=model,
            messages=messages,
            references=[],  # No references for individual models
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return {"output": output}
    except Exception as e:
        print(f"Error in process_fn: {str(e)}")
        return None

def bot(history, temperature, max_tokens, rounds):
    user_message = history[-1][0]
    bot_message = ""
    log_output = ""
    
    output, log_output = moa_generate(user_message, [msg for pair in history[:-1] for msg in pair], MODEL_AGGREGATE, default_reference_models, temperature, max_tokens, rounds)
    
    for chunk in output:
        if isinstance(chunk, dict):
            if 'content' in chunk:
                bot_message += chunk['content']
            elif 'choices' in chunk and len(chunk['choices']) > 0:
                if 'text' in chunk['choices'][0]:
                    bot_message += chunk['choices'][0]['text']
                elif 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    bot_message += chunk['choices'][0]['delta']['content']
        else:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'text'):
                    bot_message += chunk.choices[0].text
                elif hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    bot_message += chunk.choices[0].delta.content
        
        history[-1][1] = bot_message
        yield history, log_output

    return history, log_output

def process_bot_response(message, history, aggregate_model, reference_models, temperature, max_tokens, rounds, multi_turn, system_prompt):
    if not multi_turn:
        history = []
    
    flat_history = [msg for pair in history for msg in pair if msg is not None]
    
    output, log_output = moa_generate(message, flat_history, aggregate_model, reference_models, temperature, max_tokens, rounds, system_prompt)
    
    bot_message = ""
    for chunk in output:
        if isinstance(chunk, dict):
            if 'content' in chunk:
                bot_message += chunk['content']
            elif 'choices' in chunk and len(chunk['choices']) > 0:
                if 'text' in chunk['choices'][0]:
                    bot_message += chunk['choices'][0]['text']
                elif 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    bot_message += chunk['choices'][0]['delta']['content']
        else:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'text'):
                    bot_message += chunk.choices[0].text
                elif hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    bot_message += chunk.choices[0].delta.content
    
    new_history = history + [[message, bot_message]]
    return new_history, log_output

def create_gradio_interface():
    # Custom theme with earth tones
    theme = gr.themes.Base(
        primary_hue="green",
        secondary_hue="stone",
        neutral_hue="gray",
        font=("Helvetica", "sans-serif"),
    ).set(
        body_background_fill="linear-gradient(to right, #2c5e1a, #4a3728)",
        body_background_fill_dark="linear-gradient(to right, #1a3c0f, #2e2218)",
        button_primary_background_fill="#4a3728",
        button_primary_background_fill_hover="#5c4636",
        block_title_text_color="#e0d8b0",
        block_label_text_color="#c1b78f",
    )

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
            
            # Mixture of Agents (MoA) Chat
            
            Welcome to the future of AI-powered conversations! This app combines multiple AI models
            to generate responses, merging their strengths for more accurate and diverse outputs.
            
            </div>
            """,
            elem_id="centered-markdown"
        )        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Model Configuration", open=True):
                    aggregate_model = gr.Dropdown(
                        choices=[MODEL_AGGREGATE, MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3],
                        value=MODEL_AGGREGATE,
                        label="Aggregate Model"
                    )
                    reference_models_box = gr.CheckboxGroup(
                        choices=[MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3],
                        value=default_reference_models,
                        label="Reference Models"
                    )
                
                with gr.Accordion("Generation Parameters", open=True):
                    temperature = gr.Slider(minimum=0, maximum=1, value=float(TEMPERATURE), label="Temperature")
                    max_tokens = gr.Slider(minimum=1, maximum=4096, step=1, value=int(MAX_TOKENS), label="Max Tokens")
                    rounds = gr.Slider(minimum=1, maximum=5, step=1, value=int(ROUNDS), label="Rounds")
                    multi_turn = gr.Checkbox(value=MULTITURN, label="Multi-turn Conversation")
                    system_prompt = gr.Textbox(
                        value="You are a helpful AI assistant.",
                        label="System Prompt",
                        lines=2
                    )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", height=400)
                msg = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter...", lines=2)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
        
        with gr.Accordion("Logs", open=False):
            logs = gr.Textbox(label="Processing Logs", lines=10)
        
        gr.Markdown(
            """
            ### How it works
            1. Your message is sent to multiple reference models.
            2. Each model generates a response.
            3. The aggregate model combines these responses to create a final output.
            4. The process repeats for the specified number of rounds.
            
            This approach allows for more nuanced and well-rounded responses!
            """
        )

        def send_message(message, history):
            return "", history + [[message, None]]

        def clear_chat():
            return None, None

        def process_bot_response(message, history, aggregate_model, reference_models, temperature, max_tokens, rounds, multi_turn, system_prompt):
            if not multi_turn:
                history = []
            
            flat_history = [msg for pair in history for msg in pair if msg is not None]
            
            output, log_output = moa_generate(message, flat_history, aggregate_model, reference_models, temperature, max_tokens, rounds, system_prompt)
            
            bot_message = ""
            for chunk in output:
                if isinstance(chunk, dict):
                    if 'content' in chunk:
                        bot_message += chunk['content']
                    elif 'choices' in chunk and len(chunk['choices']) > 0:
                        if 'text' in chunk['choices'][0]:
                            bot_message += chunk['choices'][0]['text']
                        elif 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                            bot_message += chunk['choices'][0]['delta']['content']
                else:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        if hasattr(chunk.choices[0], 'text'):
                            bot_message += chunk.choices[0].text
                        elif hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            bot_message += chunk.choices[0].delta.content
            
            new_history = history + [[message, bot_message]]
            return new_history, log_output

        msg.submit(process_bot_response, [msg, chatbot, aggregate_model, reference_models_box, temperature, max_tokens, rounds, multi_turn, system_prompt], [chatbot, logs])
        send_btn.click(process_bot_response, [msg, chatbot, aggregate_model, reference_models_box, temperature, max_tokens, rounds, multi_turn, system_prompt], [chatbot, logs])

        clear_btn.click(clear_chat, outputs=[chatbot, logs])

    return demo

# Create the Gradio interface
demo = create_gradio_interface()

if __name__ == "__main__":
    demo.queue()
    demo.launch()
