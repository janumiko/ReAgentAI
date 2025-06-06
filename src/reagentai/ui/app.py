import functools

import gradio as gr

from src.reagentai.common.client import LLMClient
from src.reagentai.constants import AVAILABLE_LLM_MODELS


# UI Creation Helper Functions
def create_settings_panel():
    with gr.Column(scale=1, min_width=250):
        gr.Markdown("### Model Settings")
        llm_model_dropdown = gr.Dropdown(
            label="Select LLM Model",
            choices=AVAILABLE_LLM_MODELS,
            value=AVAILABLE_LLM_MODELS[0],
        )
        token_usage_display = gr.Number(
            label="Total Token Usage",
            value=0,
            precision=0,
            interactive=False,
            visible=True,
        )
    return llm_model_dropdown, token_usage_display


def create_chat_interface():
    with gr.Column(scale=3):
        gr.Markdown("### Chat & Results")
        chatbot_display = gr.Chatbot(
            type="messages",
            label="Conversation",
            bubble_full_width=False,
            height=800,
        )
        chat_input = gr.MultimodalTextbox(
            placeholder="Type your query here...",
            file_count="multiple",
            sources=None,
            show_label=False,
        )
    return chatbot_display, chat_input


# Event Handler Functions
def add_user_message_to_history(chat_history: list, user_input: dict):
    """
    Adds user message (text and/or files) to the chat history.
    Files are added first, then the text message if it exists.
    Disables the input textbox temporarily.
    """
    for file_path in user_input.get("files", []):
        chat_history.append({"role": "user", "content": {"path": file_path}})
    if user_input.get("text"):
        chat_history.append({"role": "user", "content": user_input["text"]})
    return chat_history, gr.MultimodalTextbox(value=None, interactive=False)


def handle_bot_response(chat_history: list, llm_client: LLMClient):
    """
    Gets LLM response, updates chat history and token usage.
    """
    user_query = chat_history[-1]["content"] if chat_history else ""
    response = llm_client.respond(user_query)
    chat_history.extend(response)
    token_used = llm_client.get_token_usage()
    return chat_history, token_used


def handle_clear_chat(llm_client: LLMClient):
    """
    Clears LLM history and resets chat display.
    """
    llm_client.clear_history()
    return [], 0


def handle_model_change(model_name: str, llm_client: LLMClient):
    """
    Sets the new LLM model in the client.
    """
    llm_client.set_model(model_name)


def re_enable_chat_input():
    """
    Re-enables the chat input textbox.
    """
    return gr.MultimodalTextbox(interactive=True)


# Main App Creation Function
def create_gradio_app(llm_client: LLMClient):
    with gr.Blocks(
        theme=gr.themes.Origin(),
    ) as demo:
        gr.Markdown(
            """
            # ReagentAI - Retrosynthesis Assistant
            """
        )

        with gr.Row():
            llm_model_dropdown, token_usage_display = create_settings_panel()
            chatbot_display, chat_input = create_chat_interface()

        # Event handling
        chat_input.submit(
            fn=add_user_message_to_history,
            inputs=[chatbot_display, chat_input],
            outputs=[chatbot_display, chat_input],
        ).then(
            fn=functools.partial(handle_bot_response, llm_client=llm_client),
            inputs=chatbot_display,
            outputs=[chatbot_display, token_usage_display],
            api_name="bot_response",
        ).then(fn=re_enable_chat_input, inputs=None, outputs=[chat_input])

        chatbot_display.clear(
            fn=functools.partial(handle_clear_chat, llm_client=llm_client),
            inputs=[],
            outputs=[chatbot_display, token_usage_display],
            api_name="clear_chat",
        )

        llm_model_dropdown.change(
            fn=functools.partial(handle_model_change, llm_client=llm_client),
            inputs=llm_model_dropdown,
            outputs=[],
        )

    return demo
