import asyncio


class StringStreamer:
    def __init__(self, stream_string):
        self.stream_string = stream_string

    def generate(self):
        output = self.stream_string.split()
        for out in output:
            yield out + ' '


class AsyncStreamer:
    def __init__(self, streamer):
        """
        Initializer method that takes a iterable object (streamer) as input.
        """
        self.streamer = streamer

    def __getattr__(self, attr):
        """
        When an attribute is not found in AsyncStreamer,
        fall back to looking it up in the streamer object.
        """
        return getattr(self.streamer, attr)

    @staticmethod
    def safe_next(iterator):
        """
        Static method that safely fetches the next item from the iterator.
        If the iterator is exhausted, it catches the StopIteration and returns None.
        """
        try:
            return next(iterator)
        except StopIteration:
            return None

    def __aiter__(self):
        """
        Method required for an object to be an asynchronous iterable.
        Returns the instance (self), meaning the object itself is the asynchronous iterable.
        """
        return self

    async def __anext__(self):
        """
        Asynchronous method to fetch the next item in the asynchronous iteration.
        Uses an executor to run safe_next which provides the next item from the streamer
        or None if there are no more items.
        If None is encountered, it raises StopAsyncIteration to signal end of iteration.
        """
        loop = asyncio.get_event_loop()
        item = await loop.run_in_executor(None, self.safe_next, self.streamer)
        if item is None:
            raise StopAsyncIteration
        else:
            return item
