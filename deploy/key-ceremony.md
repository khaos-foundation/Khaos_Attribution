# The key ceremony

The register countersigns every record with one Ed25519 key. Until the ceremony it runs on a development key and says so on every page. The ceremony makes the Guardians' key, puts it on the server, and records in the ledger who was there.

What the code can and cannot do, said plainly: the service signs from a PEM file it reads at start. It cannot sign from a hardware token. So the key is made on a machine that is offline for the whole ceremony and is written to removable media, never to that machine's own disk. The plan's "hardware key" is, for now, a sealed USB stick.

## The short ceremony

Two of the Foundation's directors, each with a YubiKey 5 (any 5-series key; the FIDO-only Security Key series cannot do this), one USB stick, one envelope, about twenty minutes. The key is made on the server, where it lives anyway. Its only backup is encrypted to both YubiKeys: either one with its PIN opens it, nothing else does. The laptop needs `brew install age age-plugin-yubikey`.

1. Each director, with their own YubiKey plugged in, sets it up once and types their own PIN:
   ```
   age-plugin-yubikey --generate
   age-plugin-yubikey --identity > yubikey-<name>.txt
   ```
   The first prints a recipient (`age1yubikey1…`). Neither the recipient nor the identity file is secret. Each director writes their PUK on the sheet; nobody else sees a PIN.
2. Make the key; the register keeps signing with the development key meanwhile:
   ```
   register-key make
   ```
   Write the key id and the hash on the sheet.
3. Lock the backup on the laptop; the key passes through memory, and only the encrypted copy is written:
   ```
   deploy/register-key-backup.sh lock register-key.age <recipient-1> <recipient-2>
   deploy/register-key-backup.sh check register-key.age yubikey-<name-1>.txt
   deploy/register-key-backup.sh check register-key.age yubikey-<name-2>.txt
   ```
   Both checks must say the copy matches the server's key. Copy `register-key.age` to the stick and to the Foundation's storage box. Seal the stick and the sheet in the envelope; both sign across the flap.
4. Switch, naming who was there:
   ```
   register-key switch "Name, Name"
   ```
   It retires the development key, signs with the new one, checks the server now says so, and writes the names and the key id into the public ledger. If the switch's check fails it puts the development key back; if only the ledger entry fails, the new key stays and the command to record it is printed.

To restore the key onto a new server, plug in either YubiKey: `age -d -i yubikey-<name>.txt register-key.age`, piped straight to the server. Each step refuses to run twice, and neither server step runs once the development setting is gone. The service itself remembers which key it served as the development key and refuses to start on it without that setting, so deleting the setting by hand never promotes it. When more Guardians are named later, the same key is kept and their custody is recorded with `register-cli key --present "Name, Name" --note "custody"`.

### The log key

The signed append-only log signs its tree heads with a second key, never the countersigning one, made and locked the same way at the same sitting:

```
register-log-key make
REGISTER_STAGED=/srv/register/data/staged-log/log-key.pem deploy/register-key-backup.sh lock log-key.age <recipient-1> <recipient-2>
REGISTER_STAGED=/srv/register/data/staged-log/log-key.pem deploy/register-key-backup.sh check log-key.age yubikey-<name>.txt
register-log-key switch "Name, Name"
```

The switch retires the development log key and, once the new one is serving, destroys it (`shred -u`; it was only ever a development key, and a key that still exists could sign heads), restarts the service (which signs the current tree with the new key), checks `/log/key` says the new key and not development, and records the key in the ledger (`register-cli key --log`). It prints the public key: commit it to the contract at `src/khaos_attribution/translog/log-key.pub`, release, and from then on `khaos-log-verify` and the `/verify-entry` page pin it. Write its key id on the sheet beside the countersigning key's.

The full ceremony below keeps the key off the server's own tools while it is made. Use it when the Guardians want that.

## The full ceremony: before

- The Guardians named and present. At least two.
- A laptop that can be started from a live Linux USB with networking off, with Python 3.12 available on it, and the contract's `[registry]` extra installed on that live system beforehand (the ceremony itself is offline).
- Three new USB sticks, labelled SERVER, BACKUP, PUBLIC.
- The register server reachable from another machine, with `REGISTER_DEV_KEY=1` still set.

## The ceremony

1. Start the offline laptop. Confirm networking is off. Mount the three sticks.
2. Make the key straight onto the SERVER stick:
   ```
   python -m register_service make-key /media/SERVER/register-key.pem
   ```
   It prints the key id and the public key. The private half is at `register-key.pem`, the public half beside it as `register-key.pub`. Nothing was written to the laptop.
3. Copy `register-key.pem` and `register-key.pub` to the BACKUP stick. Copy `register-key.pub` alone to the PUBLIC stick. Compare the three copies by hash:
   ```
   sha256sum /media/*/register-key.*
   ```
4. Write the key id and the public key's sha256 on the ceremony sheet. Every Guardian signs the sheet.
5. Seal the BACKUP stick in an envelope signed across the flap by every Guardian. It is opened only to restore the server's key, by two Guardians together.
6. Shut the laptop down. The live system leaves nothing behind.

## On the server, the same day

1. Place the key, as root:
   ```
   install -o register -g register -m 600 /media/SERVER/register-key.pem /srv/register/data/register-key.pem
   install -o register -g register -m 644 /media/SERVER/register-key.pub /srv/register/data/register-key.pub
   sed -i '/^REGISTER_DEV_KEY=/d' /etc/register.env
   systemctl restart register.service
   ```
2. Check the page no longer says development, and the key id matches the sheet:
   ```
   curl -fsS https://register.khaos.dev/api/public-key
   ```
3. Record the ceremony as the ledger's entry:
   ```
   register-cli key --present "Name, Name" --note "first ceremony"
   ```
   The entry carries the key id, the public key's hash and the names. It is in the public ledger from then on.
4. Wipe the SERVER stick.

## After

- Every Space pins the public key from the PUBLIC stick (or from `/api/public-key`, compared against the sheet) in its `[register] public_key_pem`. An exposed Space serves nothing countersigned by the development key from then on, so anything published before the ceremony is published again.
- Every Workshop that enrolled on the development key runs `scripts/register.py pin` to pin the new key.

## What would change this

A signing step that can use a hardware token (PKCS#11 or an agent) would let the private key never exist as a file at all. It is not written. Until it is, the BACKUP envelope is the only copy of the key that is not on the server.
