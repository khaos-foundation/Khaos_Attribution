"""The log's construction checked offline: RFC 6962 tree hashes and proofs against the Certificate Transparency
vectors, every proof over sizes 1..64, signed heads, and RFC 8785 canonicalisation."""
import json
import struct
import sys
import unicodedata

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("rfc8785")

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402

CT_INPUTS = [b"", b"\x00", b"\x10", b"\x20\x21", b"\x30\x31", b"\x40\x41\x42\x43", bytes(range(0x50, 0x58)), bytes(range(0x60, 0x70))]
CT_ROOTS = ["6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
            "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
            "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
            "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
            "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
            "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328"]


def _leaves(n):
    return [T.leaf_hash(f"entry {i}".encode()) for i in range(n)]


def test_tree_hashes_and_proofs_match_the_certificate_transparency_vectors():
    """Leaf and node prefixes, the split for sizes that are not powers of two, and proof order, as RFC 6962 has them."""
    tree = T.Tree.from_leaves([T.leaf_hash(x) for x in CT_INPUTS])
    assert [tree.root(n).hex() for n in range(1, 9)] == CT_ROOTS
    assert [p.hex()[:8] for p in tree.inclusion(0, 8)] == ["96a296d2", "5f083f0a", "6b47aaf2"]
    assert [p.hex()[:8] for p in tree.consistency(6, 8)] == ["0ebc5d34", "ca854ea1", "d37ee418"]
    assert [p.hex()[:8] for p in tree.consistency(2, 5)] == ["5f083f0a", "bc1a0643"]


def test_every_inclusion_proof_verifies_offline_and_a_wrong_leaf_index_or_root_does_not():
    """Inclusion proves from the proof and the root alone, for every leaf of every size 1..64."""
    leaves = _leaves(64)
    tree = T.Tree.from_leaves(leaves)
    for n in range(1, 65):
        root = tree.root(n)
        for i in range(n):
            T.verify_inclusion(leaves[i], i, n, tree.inclusion(i, n), root)
    path = tree.inclusion(5, 13)
    for bad in ((leaves[6], 5, 13, path, tree.root(13)), (leaves[5], 6, 13, path, tree.root(13)),
                (leaves[5], 5, 13, path, tree.root(12)), (leaves[5], 5, 13, path[:-1], tree.root(13)),
                (leaves[5], 5, 13, path + [bytes(32)], tree.root(13)), (leaves[5], 5, 4, path, tree.root(13))):
        with pytest.raises(T.LogError):
            T.verify_inclusion(*bad)


def test_consistency_proofs_verify_between_any_two_sizes_up_to_64_and_a_rewritten_prefix_fails():
    leaves = _leaves(64)
    tree = T.Tree.from_leaves(leaves)
    for m in range(1, 65):
        for n in range(m, 65):
            T.verify_consistency(m, n, tree.root(m), tree.root(n), tree.consistency(m, n))
    forged = T.Tree.from_leaves(leaves[:3] + [T.leaf_hash(b"changed")] + leaves[4:])
    for m in range(4, 64):
        with pytest.raises(T.LogError):
            T.verify_consistency(m, 64, tree.root(m), forged.root(64), forged.consistency(m, 64))
    proof = tree.consistency(5, 9)
    with pytest.raises(T.LogError):
        T.verify_consistency(5, 9, tree.root(5), tree.root(9), [proof[0]] + [bytes(32)] + proof[2:])


def test_altering_any_past_entry_breaks_verification_against_every_later_head():
    """Change one entry: its old proof fails against every head that includes it, and no proof leads the new leaf there."""
    key = R.SigningKey.generate()
    leaves = _leaves(20)
    tree = T.Tree.from_leaves(leaves)
    heads = [T.sign_head(n, tree.root(n), key) for n in range(1, 21)]
    for victim in (0, 7, 19):
        changed = leaves[:victim] + [T.leaf_hash(b"rewritten")] + leaves[victim + 1:]
        forged = T.Tree.from_leaves(changed)
        for head in heads[victim:]:
            n = head["tree_size"]
            root = bytes.fromhex(head["root_sha256"])
            with pytest.raises(T.LogError):
                T.verify_inclusion(changed[victim], victim, n, forged.inclusion(victim, n), root)
            with pytest.raises(T.LogError):
                T.verify_inclusion(changed[victim], victim, n, tree.inclusion(victim, n), root)
            assert forged.root(n) != root


def test_a_head_is_ed25519_over_the_canonical_form_of_its_five_fields_and_nothing_else_verifies():
    key, other = R.SigningKey.generate(), R.SigningKey.generate()
    head = T.sign_head(7, bytes(range(32)), key, timestamp="2026-10-27T12:00:00Z")
    assert T.head_signed_bytes(head) == (b'{"log_id":"guild-register","root_sha256":"' + bytes(range(32)).hex().encode()
                                         + b'","timestamp":"2026-10-27T12:00:00Z","tree_size":7,"version":"guild.register.log.head/1"}')
    T.verify_head(head, key.public_pem())
    for bad in ({**head, "tree_size": 8}, {**head, "root_sha256": "0" * 64}, {**head, "timestamp": "2026-10-28T12:00:00Z"}):
        with pytest.raises(T.LogError, match="signature"):
            T.verify_head(bad, key.public_pem())
    with pytest.raises(T.LogError, match="pinned"):
        T.verify_head(head, other.public_pem())


def test_canonicalisation_is_stable_across_key_order_and_follows_rfc_8785():
    """Key order and whitespace never change the bytes; the RFC's own example and number vectors hold."""
    a = T.make_entry("admit", time="2026-10-27T12:00:00Z", salt="0" * 32, serial=65536, record_sha256="a" * 64)
    b = dict(reversed(list(a.items())))
    assert T.jcs(a) == T.jcs(b) and T.entry_leaf_hex(a) == T.entry_leaf_hex(b)
    example = json.loads(r'''{"numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001], "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/", "literals": [null, true, false]}''')
    assert T.jcs(example) == (b'{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"strin'
                              b'g":"\xe2\x82\xac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}')
    vectors = {"0000000000000000": "0", "8000000000000000": "0", "0000000000000001": "5e-324", "8000000000000001": "-5e-324",
               "7fefffffffffffff": "1.7976931348623157e+308", "4340000000000000": "9007199254740992",
               "4430000000000000": "295147905179352830000", "44b52d02c7e14af6": "1e+23", "444b1ae4d6e2ef50": "1e+21",
               "3eb0c6f7a0b5ed8c": "9.999999999999997e-7", "3eb0c6f7a0b5ed8d": "0.000001",
               "41b3de4355555553": "333333333.3333332", "becbf647612f3696": "-0.0000033333333333333333",
               "43143ff3c1cb0959": "1424953923781206.2"}
    for bits, text in vectors.items():
        assert T.jcs(struct.unpack(">d", bytes.fromhex(bits))[0]) == text.encode(), bits
    with pytest.raises(Exception):
        T.jcs(float("nan"))


def test_canonicalisation_does_not_normalise_unicode_so_callers_send_nfc():
    """RFC 8785 keeps strings as given: the composed and decomposed forms of one name are two entries."""
    composed, decomposed = "Zoë", unicodedata.normalize("NFD", "Zoë")
    assert composed != decomposed and unicodedata.normalize("NFC", decomposed) == composed
    one = T.make_entry("revoke", time="2026-10-27T12:00:00Z", salt="0" * 32, reason=composed)
    two = {**one, "reason": decomposed}
    assert T.entry_leaf_hex(one) != T.entry_leaf_hex(two)


def test_an_entry_carries_a_fresh_128_bit_salt_so_a_withheld_body_cannot_be_guessed():
    a, b = T.make_entry("admit", serial=1), T.make_entry("admit", serial=1)
    assert len(a["salt"]) == 32 and a["salt"] != b["salt"] and T.entry_leaf_hex(a) != T.entry_leaf_hex(b)
    with pytest.raises(ValueError):
        T.make_entry("edit")
