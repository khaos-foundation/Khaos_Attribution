"""The installers' one door to an engine: its licence stated from the descriptor, accepted once and recorded,
then the engine's own provision slot when it has one; --check changes nothing."""
import json

from khaos_attribution.engines import provision as P


def test_the_statement_is_the_descriptors_and_names_the_gated_weights():
    s = P.statement("stable-audio-3")
    assert "Stability AI Community License" in s and "open base repository" in s and "is gated" not in s, "no account is needed"
    assert P.statement("mock").startswith("mock-engine 1.0 is under the Khaos Free Use Licence.")


def test_acceptance_is_recorded_once_and_check_refuses_without_it(tmp_path, capsys):
    assert P.main(["mock", "--check", "--marker-dir", str(tmp_path)]) == 1
    assert "has not been accepted" in capsys.readouterr().out and not list(tmp_path.iterdir())
    assert P.main(["mock", "--accept-licence", "--marker-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "accepted by --accept-licence" in out and "needs no provisioning" in out
    rec = json.loads(P.marker_path(tmp_path, "mock").read_text())
    assert rec["engine"] == "mock" and rec["licence"] == "Khaos Free Use Licence" and P.accepted(tmp_path, "mock")
    assert P.main(["mock", "--check", "--marker-dir", str(tmp_path)]) == 0
    assert "accepted earlier" in capsys.readouterr().out


def test_an_engine_with_a_provision_slot_runs_it_after_the_licence(tmp_path, monkeypatch, capsys):
    from khaos_attribution.engines.sa3 import provision as SP
    seen = {}

    def fake_provision(repo_dir, weights_dir, *, device, check_only):
        seen.update(repo_dir=repo_dir, weights_dir=weights_dir, device=device, check_only=check_only)
        r = SP.Report()
        r.bad("Stable Audio 3 code not cloned")
        return r
    monkeypatch.setattr(SP, "provision", fake_provision)
    P.record_acceptance(tmp_path, "stable-audio-3")
    code = P.main(["stable-audio-3", "--check", "--marker-dir", str(tmp_path), "--repo-dir", str(tmp_path / "sa3"), "--device", "cuda"])
    assert code == 1 and seen["check_only"] is True and seen["device"] == "cuda" and seen["repo_dir"] == tmp_path / "sa3"
    assert "CHECK FAILED" in capsys.readouterr().out
