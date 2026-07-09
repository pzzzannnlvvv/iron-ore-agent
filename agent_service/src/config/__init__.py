import os
from dotenv import load_dotenv

app_env = os.environ.get("XMSCHAIN_AGENT_ENV", "dev")
assert app_env in ("dev", "prod", "test", "local"), (
    f'Invalid APP_ENV environment value: "{app_env}"'
)

load_dotenv(f".env.{app_env}")
