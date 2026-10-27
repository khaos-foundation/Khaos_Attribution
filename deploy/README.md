# The Guild Register on its own server

One small Ubuntu 24.04 server, SQLite, a few files. `register-server.sh` does the whole setup as root and can be run again after a change of tag.

```
ssh root@<server>
git clone https://github.com/khaos-foundation/Khaos_Attribution.git /root/Khaos_Attribution
REGISTER_DOMAIN=register.khaos.dev REGISTER_TAG=v0.27.0 bash /root/Khaos_Attribution/deploy/register-server.sh
```

What it leaves behind:

- `/srv/register/`: the checkout at the tag, the venv (the `registry` and `service` extras only), `data/` (the two databases, the mirror files, the key), `backups/`.
- `/etc/register.env`: the service's settings. `REGISTER_DEV_KEY=1` until the ceremony, with `REGISTER_DEV_KEY_PUBLIC=1` beside it: a development key serves behind the proxy only when the deploy says so (binding loopback keeps nothing private once Caddy fronts it).
- `register.service` on loopback port 8912; Caddy on 80 and 443 with the certificate for the domain; the firewall open on 22, 80 and 443.
- `register-backup.timer`: nightly at 03:30, a consistent copy of both databases and the mirror into `backups/`, kept 30 days, sent to `BACKUP_TARGET` over ssh when one is set in `/etc/register.env` (a Hetzner Storage Box takes rsync over ssh on port 23). The key is never in a backup.
- `register-restore.sh <archive>`: the way back, ending with the ledger and log check.
- `register-log-stamp.timer` (daily, 02:10) and `register-log-upgrade.timer` (four times a day): the newest root of the signed append-only log stamped with OpenTimestamps, and pending receipts completed once the calendars have them confirmed. Both run `register-cli log-stamp` / `log-upgrade` and write the receipts into `mirror/log/anchors/`. A calendar being down only delays an anchor.

Accounts and invites are made on the server with `register-cli account <name>` and `register-cli invite <name> --role workshop|space` (the register's commands as the service user, with its settings). An invite is a one-time string handed to one machine.

## Updating

Update the register by re-running `register-server.sh` with the new `REGISTER_TAG`, never by checking out a tag and restarting by hand. The script also brings `/etc/register.env` up to what the new release needs: from 0.37.0 a development key serves behind Caddy only with `REGISTER_DEV_KEY_PUBLIC=1` beside `REGISTER_DEV_KEY=1`, and a settings file without that line makes the service refuse to start.

## The key

The service countersigns with the Ed25519 key at `REGISTER_KEY_FILE`, a PEM file it reads at start. With `REGISTER_DEV_KEY=1` it makes one if none is there and says on every page that nothing it signs is a Guild record. The ceremony replaces that file with the Foundation's key and removes the line: the short version is `register-key make`, then `register-key switch "Name, Name"` (see `key-ceremony.md`). The code signs from a file, not from a hardware token: a key that must never touch a disk needs a signing step the service does not have yet.

## The log

The register keeps a signed append-only log of every admission and strike (README, "Verify an entry in the Guild Register"). It lives in `register.sqlite3` beside the records, so the nightly backup carries it. Its static export is `data/mirror/log/` (served at `/mirror/log/`, every body at once) and `data/mirror-public/log/` (the same with each new body held back `REGISTER_LOG_PUBLIC_DELAY_DAYS`, default and at most 7); `register-mirror` pushes the second to the public repository, so a body withheld within the delay never enters its git history, while one withheld later stays there. On the first start after the update the service appends a `genesis` entry (the ledger's tip and length) and admits every existing record; `register-cli check` re-derives the whole log from its bodies, checks each head, and compares the database with every published copy. Warden machines are enrolled like any other: `register-cli invite <account> --role warden`; each entry they append names their machine and has a ledger row. The operator can withhold a body with `register-cli log-withhold <leaf hash> "<reason>"`. `register-log-stamp` appends a `heartbeat` entry every day (its time, nothing else), so verifiers can tell a quiet log from a frozen mirror (they warn past 48 hours); every size has exactly one signed head, and the delay of a body is counted from the head that first held it.

At start the service compares its database with the published log: the two exports, the public repository's clone at `/srv/register/ledger-mirror`, and `REGISTER_LOG_WITNESSES` (by default the public repository's raw address, `https://raw.githubusercontent.com/guild-register/ledger/main`). It fails closed: unless some copy it can read vouches for the database, it appends nothing. A database that is behind (restored from a backup older than the last publish) or apart from it appends nothing and answers 503 to every change, and `/health` says why. `register-restore.sh` keeps the published log in place for exactly this. `register-cli log-recover` puts the missing entries back verbatim from the published bodies when they reproduce the published signed root, and names any record the log holds that the database lost (its Workshop publishes it again from its outbox); then restart the service. An entry whose body was never published (withheld, or still inside the delay) is put back as its leaf alone, and a withhold entry then says its body was lost; a record the log admitted keeps its serial and countersignature when its Workshop publishes it again. When the published leaves do not reproduce the signed root, or this database is not a prefix of them, it declares an INCIDENT and changes nothing: the register stays closed for the wardens to decide. `register-restore.sh` clones the public repository afresh (when `LEDGER_REPO` is set) before the service starts.

**The very first start.** A register whose log is empty opens by itself when every copy it asks answers that nothing was published there yet (no `log/` folder, or a 404 from the public repository). If the public address cannot be reached at all (no network, a firewall), the register cannot know it is the first and stays closed: make it reachable, or, after checking by hand that the public repository holds no `log/` folder, start once with `REGISTER_LOG_UNWITNESSED=1` in `/etc/register.env` and remove the line after the first start (from then on the server's own export vouches for it).

The log signs its heads with its own key at `REGISTER_LOG_KEY_FILE`. `REGISTER_LOG_DEV_KEY=1` lets a development log key be made, exactly as for the countersigning key; its ceremony is `register-log-key make`, then `register-log-key switch "Name, Name"` (`key-ceremony.md`). The switch signs the current tree afresh with the Guild's key, which commits to every earlier entry, and prints the public half to commit at `src/khaos_attribution/translog/log-key.pub`.
