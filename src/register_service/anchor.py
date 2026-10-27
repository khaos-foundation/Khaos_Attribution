"""Anchoring the log's root in time: a daily stamp of the newest root and an upgrade of pending receipts.
Behind `Timestamper` so the backend can be swapped or dropped; reading the register never depends on it.
The OpenTimestamps backend submits the root hash itself as the digest: `ots verify -d <root> <size>.ots` checks it.
"""

from __future__ import annotations

from khaos_attribution import translog as T

CALENDARS = ("https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org",
             "https://a.pool.eternitywall.com", "https://ots.btc.catallaxy.com")
MIN_CALENDARS = 2


class Timestamper:
    """stamp(root) gives a receipt; upgrade(receipt) gives the completed receipt, or None while it is still pending."""

    name = "none"

    def stamp(self, root: bytes) -> bytes:
        raise NotImplementedError

    def upgrade(self, receipt: bytes) -> bytes | None:
        raise NotImplementedError


class OpenTimestamps(Timestamper):
    name = "opentimestamps"

    def __init__(self, calendars=CALENDARS, timeout: float = 15.0):
        self.calendars, self.timeout = tuple(calendars), timeout

    def stamp(self, root: bytes) -> bytes:
        from opentimestamps.calendar import RemoteCalendar  # noqa: PLC0415
        from opentimestamps.core.op import OpSHA256  # noqa: PLC0415
        from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp  # noqa: PLC0415
        stamp, taken, errors = Timestamp(root), 0, []
        for url in self.calendars:
            try:
                stamp.merge(RemoteCalendar(url).submit(root, timeout=self.timeout))
                taken += 1
            except Exception as e:  # noqa: BLE001 — one calendar down is not a failed stamp
                errors.append(f"{url}: {e}")
        if taken < MIN_CALENDARS:
            raise RuntimeError(f"only {taken} calendar(s) took the root: {'; '.join(errors)}")
        return _serialise(DetachedTimestampFile(OpSHA256(), stamp))

    def upgrade(self, receipt: bytes) -> bytes | None:
        from opentimestamps.calendar import DEFAULT_CALENDAR_WHITELIST, RemoteCalendar  # noqa: PLC0415
        from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation  # noqa: PLC0415
        detached = _deserialise(receipt)
        for msg, att in list(detached.timestamp.all_attestations()):
            if not isinstance(att, PendingAttestation) or att.uri not in DEFAULT_CALENDAR_WHITELIST:
                continue
            try:
                found = RemoteCalendar(att.uri).get_timestamp(msg, timeout=self.timeout)
            except Exception:  # noqa: BLE001 — not yet confirmed, or the calendar is away; tried again next run
                continue
            _merge_at(detached.timestamp, msg, found)
        done = any(isinstance(a, BitcoinBlockHeaderAttestation) for _, a in detached.timestamp.all_attestations())
        return _serialise(detached) if done else None


def _merge_at(stamp, msg: bytes, found) -> None:
    """Merges the calendar's answer into the sub-timestamp that commits to `msg`."""
    if stamp.msg == msg:
        stamp.merge(found)
        return
    for sub in stamp.ops.values():
        _merge_at(sub, msg, found)


def _serialise(detached) -> bytes:
    from opentimestamps.core.serialize import BytesSerializationContext  # noqa: PLC0415
    ctx = BytesSerializationContext()
    detached.serialize(ctx)
    return ctx.getbytes()


def _deserialise(receipt: bytes):
    from opentimestamps.core.serialize import BytesDeserializationContext  # noqa: PLC0415
    from opentimestamps.core.timestamp import DetachedTimestampFile  # noqa: PLC0415
    return DetachedTimestampFile.deserialize(BytesDeserializationContext(bytes(receipt)))


def stamp_latest(store, timestamper: Timestamper) -> dict | None:
    """The newest signed head's root, stamped once; None when there is no head or it is already stamped."""
    head = store.log.head()
    if head is None:
        return None
    if store.db.execute("SELECT 1 FROM log_anchors WHERE tree_size = ?", (head["tree_size"],)).fetchone():
        return None
    receipt = timestamper.stamp(bytes.fromhex(head["root_sha256"]))
    with store._tx():
        store.db.execute("INSERT INTO log_anchors (tree_size, root_sha256, backend, stamped_at, receipt, state) VALUES (?, ?, ?, ?, ?, 'pending')",
                         (head["tree_size"], head["root_sha256"], timestamper.name, T.now_rfc3339(), receipt))
    return {"tree_size": head["tree_size"], "root_sha256": head["root_sha256"], "state": "pending"}


def upgrade_pending(store, timestamper: Timestamper) -> list:
    """Every pending receipt asked again; the completed ones are stored and returned."""
    done = []
    for a in store.db.execute("SELECT * FROM log_anchors WHERE state = 'pending' AND backend = ? ORDER BY tree_size",
                              (timestamper.name,)).fetchall():
        upgraded = timestamper.upgrade(bytes(a["receipt"]))
        if upgraded is None:
            continue
        with store._tx():
            store.db.execute("UPDATE log_anchors SET receipt = ?, state = 'complete', completed_at = ? WHERE tree_size = ?",
                             (upgraded, T.now_rfc3339(), a["tree_size"]))
        done.append(a["tree_size"])
    return done
