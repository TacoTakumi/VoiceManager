"""Server-owned turn counter for conversational turn boundaries.

One TurnTracker per WebSocket connection. It owns the turn counter and is the single
callable place a turn is opened -- so the turn_start control frame handler is a
thin delegation to advance(), and a future server-side barge-in can open a turn from the
STT path with no client message at all. It never awaits and holds no I/O.

The counter answers one question for every outbound transcription frame: which turn does
this utterance BELONG TO? The answer is the turn that was open when the utterance's audio
was RECEIVED, not the turn current at send time -- so a final that crosses a turn boundary
(e.g. a nemotron endpoint-hold confirming after the next turn_start) keeps the OLD id.
The tracker separates those two clocks:

  * current  -- the turn now open (advance() bumps it).
  * received -- the turn open when the audio currently producing frames arrived.
                mark_received() snapshots current into it; advance() never touches it.

stamp() tags a frame with `received`, so tagging is decoupled from when the frame is sent.
"""
from typing import Callable

from voicemanager.protocol import TranscriptionMessage


class TurnTracker:
    """Per-connection turn counter that stamps turn_id on transcription frames.

    clear_partial, if given, is invoked by advance() to drop the STT pipeline's
    uncommitted live partial when a turn opens. An engine that holds no live
    partial passes nothing -- advance() then only bumps the counter.
    """

    def __init__(self, clear_partial: Callable[[], None] | None = None):
        self._current = 0            # the turn now open
        self._received = 0           # turn open when the in-flight audio was received
        self._clear_partial = clear_partial

    @property
    def current(self) -> int:
        """The turn now open (the server-owned counter's value)."""
        return self._current

    @property
    def received(self) -> int:
        """The turn open when the audio currently producing frames was received."""
        return self._received

    def mark_received(self) -> None:
        """Snapshot the open turn as the turn owning audio received now (feed time).

        Called once per inbound audio frame, before that frame is transcribed, so
        frames it produces are tagged with the turn open at receive time.
        """
        self._received = self._current

    def advance(self) -> int:
        """Open a new turn: bump the counter and clear the STT live partial.

        This is THE turn-advance method -- the turn_start frame handler and any future
        server-side barge-in call it rather than reimplementing the counter. It does not
        touch `received`, so audio already in flight keeps the turn it was received under.
        Returns the id of the newly opened turn.
        """
        self._current += 1
        if self._clear_partial is not None:
            self._clear_partial()
        return self._current

    def stamp(self, msg: TranscriptionMessage) -> TranscriptionMessage:
        """Tag a transcription frame with the turn its audio belongs to.

        Uses `received` (receive-time turn), NOT `current` (send-time turn), and returns
        the same message for convenient inline use: `await send(tracker.stamp(msg))`.
        """
        msg.turn_id = self._received
        return msg
