"""The register's commands: serve, and the operator's account and invite management.
    python -m register_service serve [--host 127.0.0.1] [--port 8912]
    python -m register_service account <name>
    python -m register_service invite <account> [--role workshop|space|warden]
    python -m register_service revoke <machine_key_id> <reason>
    python -m register_service check
    python -m register_service verify-key <holder> --kind authority|research|media    a key shown once
    python -m register_service verify-key <holder> --revoke
    python -m register_service make-key <path>            a new countersigning key, on the ceremony's machine
    python -m register_service key --present "A, B, C" [--log]    the ledger's record of a key the service now signs with
    python -m register_service log-withhold <leaf_hash> <reason>    a body out of public view; proofs unchanged
    python -m register_service log-stamp | log-upgrade    the daily anchor of the log's root, and its upgrade
"""

import argparse
import sys

from register_service.app import DEFAULT_PORT, check_dev_key_exposure, create_app, settings
from register_service.store import Store


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub.add_parser("account").add_argument("name")
    i = sub.add_parser("invite"); i.add_argument("account"); i.add_argument("--role", choices=("workshop", "space", "warden"), default="workshop")
    r = sub.add_parser("revoke"); r.add_argument("key_id"); r.add_argument("reason")
    sub.add_parser("check")
    v = sub.add_parser("verify-key"); v.add_argument("holder"); v.add_argument("--kind", default="research"); v.add_argument("--revoke", action="store_true")
    sub.add_parser("make-key").add_argument("path")
    k = sub.add_parser("key"); k.add_argument("--present", required=True, help="who was there, comma-separated"); k.add_argument("--note", default="")
    k.add_argument("--log", action="store_true", help="the log key, not the countersigning key")
    w = sub.add_parser("log-withhold"); w.add_argument("leaf_hash"); w.add_argument("reason")
    sub.add_parser("log-stamp")
    sub.add_parser("log-upgrade")
    a = p.parse_args(argv)
    cfg = settings()
    if a.cmd == "make-key":
        from pathlib import Path  # noqa: PLC0415
        from khaos_attribution.registry import SigningKey  # noqa: PLC0415
        path = Path(a.path)
        if path.exists():
            raise SystemExit(f"{path} exists; a key is made once")
        path.parent.mkdir(parents=True, exist_ok=True)
        key = SigningKey.load_or_create(path)
        print(f"key id {key.key_id}\n{key.public_pem()}", end="")
        return 0
    if a.cmd == "verify-key":
        from register_service import verify_keys  # noqa: PLC0415
        if a.revoke:
            print(f"revoked {verify_keys.revoke(cfg, a.holder)} key(s) of {a.holder}")
        else:
            print(f"{verify_keys.issue(cfg, a.holder, a.kind)}\n(shown once; give it to {a.holder})")
        return 0
    if a.cmd == "serve":
        import uvicorn  # noqa: PLC0415
        check_dev_key_exposure(cfg, a.host)
        # One worker by design: the serial and the ledger chain are kept under one process's lock.
        uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="info")
        return 0
    store = Store(cfg["data"] / "register.sqlite3")
    if a.cmd == "account":
        print(f"account {a.name}: id {store.create_account(a.name)}")
    elif a.cmd == "invite":
        print(store.create_invite(a.account, a.role))
    elif a.cmd == "revoke":
        store.revoke_machine(a.key_id, a.reason); print("revoked")
    elif a.cmd == "key":
        from register_service.app import load_log_key, load_register_key  # noqa: PLC0415
        if cfg["log_dev_key" if a.log else "dev_key"]:
            raise SystemExit(f"{'REGISTER_LOG_DEV_KEY' if a.log else 'REGISTER_DEV_KEY'} is set: a development key is never "
                             "recorded as the Guild's")
        key = load_log_key(cfg) if a.log else load_register_key(cfg)
        note = ("log key" + (f": {a.note}" if a.note else "")) if a.log else a.note
        store.record_key(key.key_id, key.public_pem(), [x.strip() for x in a.present.split(",") if x.strip()], note)
        print(f"recorded {'log ' if a.log else ''}key {key.key_id}")
    elif a.cmd == "log-withhold":
        from register_service.app import load_log_key, log_key_doc  # noqa: PLC0415
        from register_service.translog import export  # noqa: PLC0415
        log_key = load_log_key(cfg)
        store.log.signer = log_key
        out = store.log_withhold(a.leaf_hash, a.reason)
        export(store.log, cfg["data"] / "mirror", log_key_doc(log_key, cfg["log_dev_key"]))
        print(f"withheld {a.leaf_hash}; the withholding is leaf {out['index']} ({out['leaf_hash']}); the mirror is rewritten")
    elif a.cmd in ("log-stamp", "log-upgrade"):
        from register_service import anchor  # noqa: PLC0415
        if a.cmd == "log-stamp":
            out = anchor.stamp_latest(store, anchor.OpenTimestamps())
            print(f"stamped the root of size {out['tree_size']}" if out else "nothing new to stamp")
        else:
            done = anchor.upgrade_pending(store, anchor.OpenTimestamps())
            print(f"completed: {done}" if done else "no receipt completed yet")
        from register_service.translog import export_anchors  # noqa: PLC0415
        export_anchors(store.log, cfg["data"] / "mirror")
    elif a.cmd == "check":
        ok = store.ledger_ok()
        pub = cfg["log_key_file"].with_suffix(".pub")
        problems = store.log.problems(pub.read_text(encoding="ascii") if pub.is_file() else None)
        print(f"ledger {'unbroken' if ok else 'BROKEN'}: {len(store.ledger())} entries, {len(store.serials())} models")
        print(f"log {'whole' if not problems else 'BROKEN'}: {store.log.size()} entries"
              + ("" if pub.is_file() else " (no log public key beside the key file; signatures not checked)"))
        for p_ in problems[:20]:
            print(f"  {p_}")
        return 0 if ok and not problems else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
