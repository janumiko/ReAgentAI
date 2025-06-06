import gradio as gr

from src.reagentai.constants import AVAILABLE_LLM_MODELS
from src.reagentai.llm.client import LLMClient


def _create_settings_panel():
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


def _create_chat_interface():
    with gr.Column(scale=3):
        gr.Markdown("### Chat & Results")
        chatbot_display = gr.Chatbot(
            type="messages",
            label="Conversation",
            bubble_full_width=False,
            height=500,
        )
        chat_input = gr.MultimodalTextbox(
            placeholder="Type your query here...",
            file_count="multiple",
            sources=["upload"],
            show_label=False,
        )
    return chatbot_display, chat_input


def _create_gradio_app(llm_client: LLMClient):
    with gr.Blocks(
        theme=gr.themes.Origin(),
    ) as demo:
        gr.Markdown(
            """
            # ReagentAI - Retrosynthesis Assistant
            """
        )

        def bot(
            chat_history: list,
        ):
            """
            Respond to user query and update chat history.
            """
            user_query = chat_history[-1]["content"] if chat_history else ""
            response = llm_client.respond(user_query)
            chat_history.append({"role": "assistant", "content": response})
            token_used = llm_client.get_token_usage()

            return chat_history, token_used

        def clear():
            """
            Clear chat history.
            """
            llm_client.clear_history()
            return [], 0

        with gr.Row():
            llm_model_dropdown, token_usage_display = _create_settings_panel()
            chatbot_display, chat_input = _create_chat_interface()

        chat_msg = chat_input.submit(
            add_message,
            [chatbot_display, chat_input],
            [chatbot_display, chat_input],
        )
        bot_msg = chat_msg.then(bot, chatbot_display, [chatbot_display, token_usage_display], api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot_display.clear(
            clear, [], [chatbot_display, token_usage_display], api_name="clear_chat"
        )

        llm_model_dropdown.change(
            lambda model: llm_client.set_model(model),
            inputs=llm_model_dropdown,
            outputs=[],
        )

    return demo


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def get_gradio_app(llm_client: LLMClient):
    """
    Get the Gradio app for ReagentAI.

    Returns:
        gr.Blocks: The Gradio app.
    """
    return _create_gradio_app(llm_client)
