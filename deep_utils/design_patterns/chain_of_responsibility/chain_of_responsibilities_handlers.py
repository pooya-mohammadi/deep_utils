from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

from deep_utils.utils.logging_utils.logging_utils import log_print


class CoRError(Exception):
    def __init__(
        self, message: str = "", code: int = 0, output: Union[None, dict] = None
    ):
        """
        This is a custom error used to break the responsibility chain that is running through the manger!
        It helps to increase codes' readability; when one sees this error realizes that the process is broke because
        of an unfinished process that's handled but not in the desired and intended format
        :param message: message send by the user, default is ""
        :param code: code send by the user, default is 0
        :param output: A dictionary that contains output.
        """
        if output is None:
            output = None
        super().__init__(message)
        self.message = message
        self.code = code
        self.output = output


class Handler(ABC):
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    """

    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        pass

    @abstractmethod
    def handle(self, request) -> dict:
        pass


class CoRHandler(Handler):
    """
    The default chaining behavior can be implemented inside a base handler
    class.
    """

    _next_handler: Handler = None

    def set_next(self, handler: Handler) -> Handler:
        """

        :param handler:
        :return:
        """
        self._next_handler = handler
        # Returning a handler from here will let us link handlers in a
        return handler

    @abstractmethod
    def handle(self, request: Any) -> dict:
        if self._next_handler:
            return self._next_handler.handle(request)

        raise NotImplementedError(
            "next_handler is missing, and the module is not implemented!"
        )


class CoRManger:
    def __init__(
        self,
        handlers: Union[Tuple[CoRHandler], List[CoRHandler]],
        internal_error_code=-1,
    ):
        """
        The default manger, which takes in a list of handlers and process it
        :param handlers: input handlers
        :param internal_error_code: This value is returned when handlers raise an unexpected error. Default of -1
        is used for internal errors, don't use it as a code in the handlers to prevent confusion.
        """
        assert handlers, "handlers can not be None or empty"
        previous_handler: Union[CoRHandler, None] = None
        for handler in handlers:
            assert isinstance(
                handler, CoRHandler
            ), "Handlers should have CoRHandler type"
            if previous_handler is not None:
                previous_handler.set_next(handler)
            previous_handler = handler

        self.handler: CoRHandler = previous_handler
        self.internal_error_code = internal_error_code

    def manage(self, request: Any, logger=None, verbose=1):
        try:
            output = self.handler.handle(request)
        except CoRError as e:
            output = dict(e.output)
            output["code"] = e.code
            output["message"] = e.message
        except BaseException as e:
            output = dict()
            output["code"] = self.internal_error_code
            output["message"] = "Internal Error: " + str(e)
        log_print(
            logger,
            f"Request code: {output['code']}, message: {output['message']}!",
            verbose=verbose,
        )
        return output
