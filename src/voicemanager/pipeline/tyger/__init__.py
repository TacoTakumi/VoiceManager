"""The Tyger turn engine: the server side of a turn model that runs in its own process.

Tyger reads the user's mic and the agent's own playback and decides when the user's
turn ended and when the user cut in over a reply. Its model runs in a sidecar from a
separate virtual environment; this package holds the server's half. ``wire`` is the
binary hop format and the event lines the two processes exchange.

This file imports nothing, so loading ``wire`` through the package costs no more
than loading the module itself.
"""
