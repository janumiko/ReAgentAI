from collections.abc import AsyncIterator
import functools

import gradio as gr
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from src.reagentai.agents.main.main_agent import MainAgent
from src.reagentai.common.typing import ChatHistory
from src.reagentai.constants import AVAILABLE_LLM_MODELS, EXAMPLE_PROMPTS
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

        gr.Markdown("### Tool Usage History")
        tool_display = gr.Chatbot(
            type="messages",
            label="Tool Usage",
            layout="panel",
            elem_id="tool_display",
        )

        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=chat_input_component,
            label="Example Prompts",
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
        tuple: A tuple containing an empty list for chat history, an empty list for tool history, and 0 for token usage.
    """

    main_agent.clear_history()
    return [], [], 0


def handle_model_change(model_name: str, main_agent: MainAgent) -> None:
    """
    Sets the new LLM model in the client.

    Args:
        model_name (str): The name of the new LLM model to set.
        main_agent (MainAgent): The main agent instance to update with the new model.
    """
    main_agent.set_model(model_name)


async def stream_from_agent(
    prompt: str, chat_history: ChatHistory, tool_history: ChatHistory, main_agent: MainAgent
) -> AsyncIterator[tuple[gr.Component, ChatHistory, ChatHistory, int]]:
    """
    Streams the response from the main agent asynchronously and updates the chat history.

    Args:
        prompt (str): The user's query to the agent.
        chat_history (ChatHistory): The current chat history.
        tool_history (ChatHistory): The current tool usage history.
        main_agent (MainAgent): The main agent instance to use for streaming.
    Yields:
        AsyncGenerator: An asynchronous generator yielding updated components and chat histories.
    """
    # Update chat history with the user's prompt
    chat_history.append({"role": "user", "content": prompt})
    yield (
        gr.Textbox(value=None, interactive=False),
        chat_history,
        gr.skip(),
        gr.skip(),
    )  # Disable input while processing

    generated_images: list[dict] = []

    async with main_agent.run_stream(prompt) as result:
        # Stream tool calls and returns
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
                            "title": output.description,
                            "status": "done",
                        }
                        gr_message = {
                            "role": "assistant",
                            "content": {"path": output.file_path},
                            "metadata": metadata,
                        }
                        generated_images.append(gr_message)

            yield gr.skip(), gr.skip(), tool_history, gr.skip()

        # Append the assistant's response to chat history
        chat_history.append({"role": "assistant", "content": ""})
        async for message in result.stream_text():
            chat_history[-1]["content"] = message
            yield gr.skip(), chat_history, gr.skip(), gr.skip()

        # Append images to chat history
        chat_history.extend(generated_images)
        yield gr.skip(), chat_history, gr.skip(), gr.skip()

    total_tokens = main_agent.get_total_token_usage()
    yield (
        gr.Textbox(interactive=True),
        gr.skip(),
        gr.skip(),
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
        css="""
        .contain { display: flex !important; flex-direction: column !important; }
        #component-0, #component-3, #component-10, #component-8  { height: 100% !important; }
        #chatbot_display { flex-grow: 1 !important; overflow: auto !important;}
        #tool_display { flex-grow: 1 !important; overflow: auto !important;}
        #col { height: calc(100vh - 112px - 16px) !important; }
        """,
    ) as demo:
        # Main app layout
        gr.Markdown(
            """
            # ReagentAI - Retrosynthesis Assistant
            """
        )
        with gr.Row(equal_height=False):
            chatbot_display, chat_input = create_chat_interface()
            llm_model_dropdown, usage_counter, tool_display = create_settings_panel(chat_input)

        # Event handling
        chat_input.submit(
            functools.partial(
                stream_from_agent,
                main_agent=main_agent,
            ),
            inputs=[chat_input, chatbot_display, tool_display],
            outputs=[chat_input, chatbot_display, tool_display, usage_counter],
        )

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
