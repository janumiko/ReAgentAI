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
            label="Conversation",
            bubble_full_width=False,
            height=500,
        )
        user_query_textbox = gr.Textbox(
            label="Your Message",
            placeholder="Ask a question, e.g., 'Explain the first step of Route 1' or press Send to analyze.",
            show_label=False,
        )
        send_button = gr.Button(
            "Send",
            variant="primary",
        )
    return chatbot_display, user_query_textbox, send_button


def _create_gradio_app(llm_client: LLMClient):
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange"),
    ) as demo:
        gr.Markdown(
            """
            # ReagentAI - Retrosynthesis Assistant
            """
        )

        def respond(
            user_query: str,
            chat_history: list,
        ):
            """
            Respond to user query and update chat history.
            """
            response = llm_client.respond(user_query)
            chat_history.append((user_query, response))
            token_used = llm_client.get_token_usage()

            return "", chat_history, token_used

        def clear():
            """
            Clear chat history.
            """
            llm_client.clear_history()
            return [], 0

        with gr.Row():
            llm_model_dropdown, token_usage_display = _create_settings_panel()
            chatbot_display, user_query_textbox, send_button = _create_chat_interface()

            chatbot_display.clear(
                clear,
                [],
                [chatbot_display, token_usage_display],
            )

            send_button.click(
                respond,
                [user_query_textbox, chatbot_display],
                [user_query_textbox, chatbot_display, token_usage_display],
            )

            llm_model_dropdown.change(
                lambda model_name: llm_client.change_model(model_name),
                [llm_model_dropdown],
                [],
            )

    return demo


def get_gradio_app(llm_client: LLMClient):
    """
    Get the Gradio app for ReagentAI.

    Returns:
        gr.Blocks: The Gradio app.
    """
    return _create_gradio_app(llm_client)
