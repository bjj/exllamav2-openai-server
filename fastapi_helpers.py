from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
import typing, json

# Helper that takes a stream of objects and streams using "Server-sent events"
SyncJsonStream = typing.Iterator[typing.Any]
AsyncJsonStream = typing.AsyncIterable[typing.Any]
JsonStream = typing.Union[AsyncJsonStream, SyncJsonStream]
class StreamingJSONResponse(StreamingResponse):
    def __init__(
            self,
            content: JsonStream,
            **kw
    ) -> None:
        async def json_iterator():
            if isinstance(content, typing.AsyncIterable):
                iter = content
            else:
                from starlette.concurrency import iterate_in_threadpool
                iter = iterate_in_threadpool(content)
        
            async for chunk in iter:
                text = json.dumps(jsonable_encoder(chunk),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=None,
                    separators=(",", ":"),
                )
                text = "data: " + text + "\n\n"
                yield text.encode("utf-8")
            yield "data: [DONE]".encode("utf-8")

        super().__init__(content=json_iterator(), media_type="text/event-stream", **kw)