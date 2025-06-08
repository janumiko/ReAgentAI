import logging

from dotenv import load_dotenv

from src.reagentai.agents.main.main_agent import create_main_agent
from src.reagentai.logging import setup_logging
from src.reagentai.ui.app import create_gradio_app

logger = logging.getLogger(__name__)


async def start_agent():
    setup_logging()
    load_dotenv()

    main_agent = create_main_agent()
    app = create_gradio_app(main_agent)

    app.launch(server_name="0.0.0.0")
