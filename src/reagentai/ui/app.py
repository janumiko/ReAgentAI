import functools

import gradio as gr
from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from src.reagentai.agents.main.main_agent import MainAgent
from src.reagentai.common.typing import ChatHistory
from src.reagentai.constants import (
    APP_CSS,
    AVAILABLE_LLM_MODELS,
    EXAMPLE_PROMPTS,
    EXAMPLES_PER_PAGE,
)
from src.reagentai.models.output import ImageOutput


# UI Creation Helper Functions
def create_settings_panel(
    chat_input_component: gr.Textbox,
) -> tuple[gr.Dropdown, gr.Number, gr.Chatbot]:
    """
    Creates the settings panel with model selection, token usage counter, and tool usage history.

    Args:
        chat_input_component (gr.Textbox): The chat input component to be used for submitting queries.
    Returns:
        tuple: A tuple containing the model dropdown, usage counter, and tool display components.
    """
    with gr.Column(scale=2, elem_id="col"):
        with gr.Tab("Settings"):
            gr.Markdown("### Model Settings")
            model_dropdown = gr.Dropdown(
                label="Select LLM Model",
                choices=AVAILABLE_LLM_MODELS,
                value=AVAILABLE_LLM_MODELS[0],
            )
            usage_counter = gr.Number(
                label="Total Token Usage",
                value=0,
                precision=0,
                interactive=False,
                visible=True,
            )

            gr.Examples(
                examples=EXAMPLE_PROMPTS,
                inputs=chat_input_component,
                label="Example Prompts",
                examples_per_page=EXAMPLES_PER_PAGE,
            )
        with gr.Tab("Tool Usage"):
            gr.Markdown("### Tool Usage History")
            tool_display = gr.Chatbot(
                type="messages",
                label="Tool Usage",
                layout="panel",
                elem_id="tool_display",
            )

    return model_dropdown, usage_counter, tool_display


def create_chat_interface() -> tuple[gr.Chatbot, gr.Textbox]:
    """
    Creates the chat interface with a chatbot display and input textbox.

    Returns:
        tuple: A tuple containing the chatbot display and chat input textbox.
    """
    with gr.Column(scale=8, elem_id="col"):
        gr.Markdown("### Chat")
        chatbot_display = gr.Chatbot(
            type="messages",
            label="Conversation",
            elem_id="chatbot_display",
        )
        chat_input = gr.Textbox(
            placeholder="Type your query here...",
            show_label=False,
            submit_btn=True,
        )
    return chatbot_display, chat_input


# Event Handler Functions
def handle_clear_chat(main_agent: MainAgent) -> tuple[list, list, int]:
    """
    Clears LLM history and resets chat display.
    Returns empty list for chat history and 0 for token usage.

    Args:
        main_agent (MainAgent): The main agent instance to clear history from.
    Returns:
        tuple: A tuple containing empty lists for chat history and tool history, and 0 for token usage.
    """

    main_agent.clear_history()
    return [], [], 0


def handle_model_change(model_name: str, main_agent: MainAgent) -> None:
    """
    Sets the new LLM model for the main agent.

    Args:
        model_name (str): The name of the new LLM model to set.
        main_agent (MainAgent): The main agent instance to update with the new model.
    """
    main_agent.set_model(model_name)


def handle_retry(chat_history: ChatHistory, retry_data: gr.RetryData) -> tuple[str, ChatHistory]:
    """
    Handles the retry action by restoring the previous prompt and chat history.

    Args:
        chat_history (ChatHistory): The current chat history.
        retry_data (gr.RetryData): The retry data containing the index of the last message.
    Returns:
        tuple: A tuple containing the previous prompt and the updated chat history.
    """
    new_history = chat_history[: retry_data.index]
    previous_prompt = chat_history[retry_data.index]["content"]
    return previous_prompt, new_history


def handle_user_prompt(
    user_prompt: str, chat_history: ChatHistory
) -> tuple[gr.Textbox, ChatHistory]:
    """
    Handles the user's prompt by appending it to the chat history.

    Args:
        user_prompt (str): The user's input prompt.
        chat_history (ChatHistory): The current chat history.
    Returns:
        tuple: A tuple containing the updated chat input component and chat history.
    """
    chat_history.append({"role": "user", "content": user_prompt})
    return gr.Textbox(interactive=False), chat_history


def run_agent(
    prompt: str,
    chat_history: ChatHistory,
    tool_history: ChatHistory,
    main_agent: MainAgent,
) -> tuple[gr.Textbox, ChatHistory, ChatHistory, int]:
    """
    Runs the main agent with the provided prompt and updates the chat history and tool usage history.

    Args:
        prompt (str): The user's query to the agent.
        chat_history (ChatHistory): The current chat history.
        tool_history (ChatHistory): The current tool usage history.
        main_agent (MainAgent): The main agent instance to use for streaming.
    Returns:
        tuple: A tuple containing the updated chat input component, chat history, tool history, and total token usage.
    """
    try:
        result = main_agent.run(prompt)
        # Append the assistant's response to chat history
        chat_history.append({"role": "assistant", "content": result.output})

        for message in result.new_messages():
            for call in message.parts:
                # Handle tool call parts
                if isinstance(call, ToolCallPart):
                    call_args = call.args_as_json_str()
                    metadata = {
                        "title": f"🛠️ Using {call.tool_name}",
                        "status": "done",
                    }
                    if call.tool_call_id is not None:
                        metadata["id"] = call.tool_call_id

                    gr_message = {
                        "role": "assistant",
                        "content": "Parameters: " + call_args,
                        "metadata": metadata,
                    }
                    tool_history.append(gr_message)

                # Handle tool return parts
                if isinstance(call, ToolReturnPart):
                    if call.tool_name in ["smiles_to_image", "route_to_image"]:
                        output: ImageOutput = call.content
                        metadata = {
                            "title": output.title,
                            "status": "done",
                        }
                        gr_message = {
                            "role": "assistant",
                            "content": {"path": output.file_path},
                            "metadata": metadata,
                        }
                        chat_history.append(gr_message)

    except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
        chat_history.append(
            {
                "role": "assistant",
                "content": f"An error occurred while processing your request:\n{str(e)}.\nTry again later.",
            }
        )
    except ModelHTTPError as e:
        chat_history.append(
            {
                "role": "assistant",
                "content": f"An error occurred while communicating with the model provider.\nPlease try again later.\nStatus Code: {e.status_code}, Error: {str(e.message)}",
            }
        )

    total_tokens = main_agent.get_total_token_usage()
    return (
        gr.Textbox(value=None, interactive=True),
        chat_history,
        tool_history,
        total_tokens,
    )  # Re-enable input after streaming


# Main App Creation Function
def create_gradio_app(main_agent: MainAgent) -> gr.Blocks:
    """
    Creates the Gradio app with the main agent and UI components.

    Args:
        main_agent (MainAgent): The main agent instance to use for the app.
    Returns:
        gr.Blocks: A Gradio Blocks instance containing the app layout and components.
    """
    with gr.Blocks(
        theme=gr.themes.Origin(),
        fill_height=True,
        css=APP_CSS,
    ) as demo:
        # Main app layout
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(
                    value="static/logo_reagent.png",
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    interactive=False,
                    container=False,
                    elem_id="logo_container",
                )

        with gr.Row():
            chatbot_display, chat_input = create_chat_interface()
            llm_model_dropdown, usage_counter, tool_display = create_settings_panel(chat_input)

        # Event handling
        chat_input.submit(
            fn=handle_user_prompt,
            inputs=[chat_input, chatbot_display],
            outputs=[chat_input, chatbot_display],
        ).then(
            functools.partial(
                run_agent,
                main_agent=main_agent,
            ),
            inputs=[chat_input, chatbot_display, tool_display],
            outputs=[chat_input, chatbot_display, tool_display, usage_counter],
        )

        chatbot_display.retry(
            fn=handle_retry,
            inputs=[chatbot_display],
            outputs=[chat_input, chatbot_display],
            api_name="retry_last_query",
        ).then(fn=main_agent.remove_last_messages)

        chatbot_display.clear(
            fn=functools.partial(
                handle_clear_chat,
                main_agent=main_agent,
            ),
            inputs=[],
            outputs=[chatbot_display, tool_display, usage_counter],
            api_name="clear_chat",
        )

        llm_model_dropdown.change(
            fn=functools.partial(
                handle_model_change,
                main_agent=main_agent,
            ),
            inputs=[llm_model_dropdown],
            outputs=[],
        )

    return demo
