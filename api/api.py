from copy import deepcopy

from ninja_extra import NinjaExtraAPI

from api import agent
from api.exceptions import ExceptionHandler


api = NinjaExtraAPI(
    title="FastShot API",
    docs_url="/docs",
)

ExceptionHandler(api).register()

api.add_router("/agent", deepcopy(agent.router))
