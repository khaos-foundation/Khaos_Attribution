"""The Guild Register's signed append-only log, in the Certificate Transparency construction (RFC 6962 / RFC 9162).
Entries are canonical JSON (RFC 8785); leaf hash SHA-256(0x00 || entry), node hash SHA-256(0x01 || left || right);
a signed tree head is Ed25519 over the canonical JSON of {log_id, root_sha256, timestamp, tree_size, version}.
Everything here is what a third party needs to check the log; the server's storage lives in register_service.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources

LOG_ID = "guild-register"
HEAD_VERSION = "guild.register.log.head/1"
ENTRY_VERSION = "guild.register.log.entry/1"
ENTRY_KINDS = ("genesis", "admit", "revoke", "withhold")
KEY_FILE = "log-key.pub"   # the Foundation's log public key, committed after the ceremony
SHA256_HEX = 64


class LogError(ValueError):
    """A proof, head or entry that does not verify; the message says which and why."""


# ---- canonical bytes and hashes ----

def jcs(doc) -> bytes:
    """RFC 8785 canonical JSON. It does not normalise unicode: callers send strings already in NFC."""
    import rfc8785  # noqa: PLC0415
    return rfc8785.dumps(doc)


def leaf_hash(entry_bytes: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + entry_bytes).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def entry_leaf_hex(entry: dict) -> str:
    return leaf_hash(jcs(entry)).hex()


def is_hex256(value) -> bool:
    return isinstance(value, str) and len(value) == SHA256_HEX and all(c in "0123456789abcdef" for c in value)


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(kind: str, *, time: str | None = None, salt: str | None = None, **fields) -> dict:
    """An entry with its time and a random 128-bit salt, so a withheld body cannot be confirmed by guessing."""
    if kind not in ENTRY_KINDS:
        raise ValueError(f"kind is one of {ENTRY_KINDS}")
    entry = {"version": ENTRY_VERSION, "kind": kind, "time": time or now_rfc3339(), "salt": salt or secrets.token_hex(16)}
    entry.update({k: v for k, v in fields.items() if v is not None})
    return entry


# ---- the tree (RFC 6962 section 2.1) ----

def _split(n: int) -> int:
    """The largest power of two smaller than n."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


class Tree:
    """Merkle tree hashes over leaves [0, size), read from the perfect subtrees `perfect(level, index)` returns."""

    def __init__(self, size: int, perfect):
        self.size, self._perfect, self._memo = size, perfect, {}

    @classmethod
    def from_leaves(cls, leaves: list) -> "Tree":
        nodes = perfect_nodes(leaves)
        return cls(len(leaves), lambda level, index: nodes[(level, index)])

    def hash(self, start: int = 0, end: int | None = None) -> bytes:
        end = self.size if end is None else end
        n = end - start
        if n == 0:
            return hashlib.sha256(b"").digest()
        if n & (n - 1) == 0 and start % n == 0:
            return self._perfect(n.bit_length() - 1, start // n)
        if (start, end) not in self._memo:
            k = _split(n)
            self._memo[(start, end)] = node_hash(self.hash(start, start + k), self.hash(start + k, end))
        return self._memo[(start, end)]

    def root(self, size: int | None = None) -> bytes:
        return self.hash(0, self.size if size is None else size)

    def inclusion(self, index: int, size: int | None = None) -> list:
        """The audit path for leaf `index` in the tree of `size` leaves, leaf to root."""
        size = self.size if size is None else size
        if not 0 <= index < size <= self.size:
            raise LogError(f"no leaf {index} in a tree of {size}")
        return self._path(index, 0, size)

    def _path(self, m: int, start: int, end: int) -> list:
        n = end - start
        if n == 1:
            return []
        k = _split(n)
        if m < k:
            return self._path(m, start, start + k) + [self.hash(start + k, end)]
        return self._path(m - k, start + k, end) + [self.hash(start, start + k)]

    def consistency(self, first: int, second: int | None = None) -> list:
        """The proof that the tree of `first` leaves is a prefix of the tree of `second`."""
        second = self.size if second is None else second
        if not 0 < first <= second <= self.size:
            raise LogError(f"no consistency proof from {first} to {second}")
        if first == second:
            return []
        return self._subproof(first, 0, second, True)

    def _subproof(self, m: int, start: int, end: int, whole: bool) -> list:
        n = end - start
        if m == n:
            return [] if whole else [self.hash(start, end)]
        k = _split(n)
        if m <= k:
            return self._subproof(m, start, start + k, whole) + [self.hash(start + k, end)]
        return self._subproof(m - k, start + k, end, False) + [self.hash(start, start + k)]


def perfect_nodes(leaves: list) -> dict:
    """Every perfect subtree's hash, keyed (level, index): what an append stores as it goes."""
    nodes = {}
    for i, leaf in enumerate(leaves):
        add_leaf(nodes, i, leaf)
    return nodes


def add_leaf(nodes: dict, index: int, leaf: bytes) -> list:
    """Adds leaf `index` and every perfect subtree it completes; returns the new (level, index, hash) triples."""
    new = [(0, index, leaf)]
    nodes[(0, index)] = leaf
    level, i, h = 0, index, leaf
    while i % 2 == 1:
        h = node_hash(nodes[(level, i - 1)], h)
        level, i = level + 1, i // 2
        nodes[(level, i)] = h
        new.append((level, i, h))
    return new


# ---- verification (RFC 9162 sections 2.1.3.2 and 2.1.4.2) ----

def verify_inclusion(leaf: bytes, index: int, size: int, path: list, root: bytes) -> None:
    """Raises LogError unless `path` proves leaf `index` is in the tree of `size` leaves with this root."""
    if not 0 <= index < size:
        raise LogError(f"leaf index {index} is outside a tree of {size}")
    fn, sn, r = index, size - 1, leaf
    for p in path:
        if sn == 0:
            raise LogError("the inclusion proof is longer than the tree")
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    if sn != 0:
        raise LogError("the inclusion proof is shorter than the tree")
    if r != root:
        raise LogError("the inclusion proof does not lead to the head's root")


def verify_consistency(first: int, second: int, first_root: bytes, second_root: bytes, proof: list) -> None:
    """Raises LogError unless `proof` shows the tree of `first` leaves is a prefix of the tree of `second`."""
    if not 0 < first <= second:
        raise LogError(f"no consistency between sizes {first} and {second}")
    if first == second:
        if proof or first_root != second_root:
            raise LogError("two heads of one size must have one root and an empty proof")
        return
    path = list(proof)
    if first & (first - 1) == 0:
        path.insert(0, first_root)
    if not path:
        raise LogError("the consistency proof is empty")
    fn, sn = first - 1, second - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            raise LogError("the consistency proof is longer than the tree")
        if fn & 1 or fn == sn:
            fr, sr = node_hash(c, fr), node_hash(c, sr)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    if sn != 0:
        raise LogError("the consistency proof is shorter than the tree")
    if fr != first_root:
        raise LogError("the consistency proof does not lead to the earlier head's root: the log was rewritten")
    if sr != second_root:
        raise LogError("the consistency proof does not lead to the later head's root")


# ---- signed tree heads ----

def head_payload(tree_size: int, root_hex: str, timestamp: str) -> dict:
    return {"version": HEAD_VERSION, "log_id": LOG_ID, "tree_size": int(tree_size), "root_sha256": root_hex, "timestamp": timestamp}


def head_signed_bytes(head: dict) -> bytes:
    """Exactly the bytes a head's signature covers: the RFC 8785 form of its five payload fields."""
    return jcs(head_payload(head["tree_size"], head["root_sha256"], head["timestamp"]))


def sign_head(tree_size: int, root: bytes, key, timestamp: str | None = None) -> dict:
    head = head_payload(tree_size, root.hex(), timestamp or now_rfc3339())
    return {**head, "key_id": key.key_id, "signature": key.sign(jcs(head))}


def verify_head(head: dict, public_pem: str) -> None:
    """Raises LogError unless the head is well formed and signed by this key."""
    from khaos_attribution import registry as R  # noqa: PLC0415
    try:
        if head["version"] != HEAD_VERSION or head["log_id"] != LOG_ID:
            raise LogError("not a head of the Guild Register's log")
        if not isinstance(head["tree_size"], int) or head["tree_size"] < 0 or not is_hex256(head["root_sha256"]):
            raise LogError("the head's size or root is malformed")
        data = head_signed_bytes(head)
    except (KeyError, TypeError) as e:
        raise LogError(f"the head is malformed: {e}") from None
    if head.get("key_id") != R.key_id_of(R.public_raw(R.load_public_pem(public_pem))):
        raise LogError(f"the head is signed by key {head.get('key_id')}, not the pinned log key")
    if not R.verify(public_pem, data, str(head.get("signature", ""))):
        raise LogError("the head's signature is not the log key's")


@dataclass
class PinnedKey:
    public_pem: str
    key_id: str


def committed_key() -> PinnedKey | None:
    """The log's public key as committed in this package, or None before the ceremony has put one here."""
    from khaos_attribution import registry as R  # noqa: PLC0415
    ref = resources.files(__name__).joinpath(KEY_FILE)
    if not ref.is_file():
        return None
    pem = ref.read_text(encoding="ascii")
    return PinnedKey(pem, R.key_id_of(R.public_raw(R.load_public_pem(pem))))


def key_id_of_pem(public_pem: str) -> str:
    """The key id of a PEM public key: the first 16 hex characters of the sha256 of its raw bytes."""
    from khaos_attribution import registry as R  # noqa: PLC0415
    return R.key_id_of(R.public_raw(R.load_public_pem(public_pem)))


def hexes(values: list) -> list:
    return [v.hex() for v in values]


def unhex(values) -> list:
    if not isinstance(values, list) or not all(is_hex256(v) for v in values):
        raise LogError("a proof is a list of sha256 hex digests")
    return [bytes.fromhex(v) for v in values]
