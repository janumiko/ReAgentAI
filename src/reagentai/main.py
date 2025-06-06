import logging

from src.reagentai.logging import setup_logging
from dotenv import load_dotenv
from src.reagentai.tools.retrosynthesis import initialize_aizynthfinder_globally
from src.reagentai.agents.main.main_agent import create_main_agent
from src.reagentai.ui.app import create_gradio_app

from src.reagentai.constants import AIZYNTHFINDER_CONFIG_PATH

logger = logging.getLogger(__name__)

def start_agent():
    setup_logging()
    load_dotenv()
    initialize_aizynthfinder_globally(
        config_path=AIZYNTHFINDER_CONFIG_PATH,
        stock="zinc",
        expansion_policy="uspto",
        filter_policy="uspto",
    )

    main_agent = create_main_agent()
    app = create_gradio_app(main_agent)

    app.launch(server_name="127.0.0.1")
