import functools
from typing import Literal

import gradio as gr

from src.reagentai.agents.main.main_agent import MainAgent
from src.reagentai.constants import AVAILABLE_LLM_MODELS, EXAMPLE_PROMPTS

ChatMessage = dict[str, str | dict[str, str]]
ChatHistory = list[ChatMessage]
UserInput = dict[str, str | list[str]]


# UI Creation Helper Functions
def create_settings_panel(
    chat_input_component: gr.MultimodalTextbox,
) -> tuple[gr.Dropdown, gr.Number]:
    with gr.Column(scale=1, min_width=200):
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

        gr.Markdown("### Examples")
        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=chat_input_component,
        )

    return llm_model_dropdown, token_usage_display


def create_chat_interface() -> tuple[gr.Chatbot, gr.MultimodalTextbox]:
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
def add_user_message_to_history(
    chat_history: ChatHistory, user_input: UserInput
) -> tuple[ChatHistory, gr.MultimodalTextbox]:
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


def handle_bot_response(
    chat_history: ChatHistory, llm_client: MainAgent, mlflow_tracker=None
) -> tuple[ChatHistory, int]:
    """
    Gets LLM response, updates chat history and token usage.
    """
    user_query: str = chat_history[-1]["content"] if chat_history else ""
    response: list[ChatMessage] = llm_client.respond(user_query)
    chat_history.extend(response)
    token_used: int = llm_client.get_token_usage()

    # Log metrics to MLflow
    if mlflow_tracker and mlflow_tracker.mlflow_enabled:
        mlflow_tracker.log_metrics(
            {"token_usage": token_used, "conversation_length": len(chat_history)}
        )

        # Log user query as param for tracking purposes
        mlflow_tracker.log_params(
            {
                f"query_{len(chat_history)}": user_query[:100]  # Truncate long queries
            }
        )

    return chat_history, token_used


def handle_clear_chat(llm_client: MainAgent) -> tuple[list[None], Literal[0]]:
    """
    Clears LLM history and resets chat display.
    Returns empty list for chat history and 0 for token usage.
    """

    llm_client.clear_history()
    return [], 0


def handle_model_change(model_name: str, llm_client: MainAgent, mlflow_tracker=None) -> None:
    """
    Sets the new LLM model in the client.
    """
    llm_client.set_model(model_name)

    # Log model change to MLflow
    if mlflow_tracker and mlflow_tracker.mlflow_enabled:
        mlflow_tracker.log_params({"llm_model": model_name})
        mlflow_tracker.set_tags({"model_changed": "true"})


def re_enable_chat_input() -> gr.MultimodalTextbox:
    """
    Re-enables the chat input textbox.
    """
    return gr.MultimodalTextbox(interactive=True)


# Main App Creation Function
def create_gradio_app(llm_client: MainAgent, mlflow_tracker=None) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Origin(),
    ) as demo:
        gr.Markdown(
            """
            # ReagentAI - Retrosynthesis Assistant
            """
        )

        with gr.Row():
            chatbot_display, chat_input = create_chat_interface()
            llm_model_dropdown, token_usage_display = create_settings_panel(chat_input)

        # Event handling
        chat_input.submit(
            fn=add_user_message_to_history,
            inputs=[chatbot_display, chat_input],
            outputs=[chatbot_display, chat_input],
        ).then(
            fn=functools.partial(
                handle_bot_response, llm_client=llm_client, mlflow_tracker=mlflow_tracker
            ),
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
            fn=functools.partial(
                handle_model_change, llm_client=llm_client, mlflow_tracker=mlflow_tracker
            ),
            inputs=llm_model_dropdown,
            outputs=[],
        )

    return demo
