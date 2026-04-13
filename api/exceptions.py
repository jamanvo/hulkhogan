import traceback

from django.http import HttpRequest
from ninja import NinjaAPI
from ninja.responses import Response


class WrongCallError(Exception):
    def __init__(self, message: str):
        self.message = message or "Wrong call"


def openai_error_handler(request: HttpRequest, exception: Exception) -> Response:
    traceback.print_exception(exception)
    return Response({"message": str(exception)}, status=429)


def server_error_handler(request: HttpRequest, exception: Exception) -> Response:
    traceback.print_exception(exception)
    return Response({"message": str(exception)}, status=500)


class ExceptionHandler:
    def __init__(self, api: NinjaAPI):
        self.api = api

    def register(self):
        self.api.exception_handler(WrongCallError)(server_error_handler)
        self.api.exception_handler(Exception)(server_error_handler)
