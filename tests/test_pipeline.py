import json

import pytest

from cemr.config import Config
from cemr.dialogue.controller import BeliefController, Claim
from cemr.evaluation.harness import evaluate
from cemr.index import MemoryIndex
from cemr.pipeline import Pipeline, build_index
from cemr.schemas import BeliefCondition, Edit, Fact, FalseMemory, GroundTruth, TaskId
from cemr.utils import contains_value, parse_clock, parse_hour_file_id, read_jsonl

from .fixtures import build_toy_dataset
from .mock_llm import MockLLM


@pytest.fixture()
def setup(tmp_path):
    root = build_toy_dataset(tmp_path / "data")
    cfg = Config()
    cfg.data.captions_glob = str(root / "captions/*.json")
    cfg.data.events_glob = str(root / "events/*.json")
    cfg.data.train_days, cfg.data.dev_days, cfg.data.test_days = (1,), (2,), (3,)
    cfg.output_dir = str(tmp_path / "out")
    cfg.data.index_dir = str(tmp_path / "out" / "index")
    index = build_index(cfg)
    mock = MockLLM(index)
    pipe = Pipeline(cfg, index, {"generator": mock, "verifier": mock, "user_sim": mock})
    return cfg, index, pipe, mock


def test_time_parsing():
    assert parse_clock("00:09:45,000") == 585.0
    assert parse_clock("00:09:43.500") == 583.5
    assert parse_hour_file_id("A1_JAKE_DAY1_11000000") == (1, 39600)


@pytest.mark.parametrize('quota', [True, False])
def test_api_failure_stops_queued_samples_without_rejection(setup, monkeypatch, quota):
    from cemr.llm_client import LLMQuotaExhausted, LLMUnavailable
    cfg, _, pipe, _ = setup
    cfg.llm.concurrency = 1
    pipe.mine()
    monkeypatch.setattr(pipe, 'preflight', lambda: None)
    calls = []
    def fail(cand, counter):
        calls.append(cand.candidate_id)
        raise (LLMQuotaExhausted if quota else LLMUnavailable)('API unavailable')
    monkeypatch.setattr(pipe, '_one', fail)
    with pytest.raises(SystemExit, match='Generation stopped'):
        pipe.generate(limit_per_task=6)
    assert len(calls) == 1
    assert not list(read_jsonl(pipe.out / 'rejects.jsonl'))
    assert (pipe.out / 'funnel.json').exists()


def test_index(setup):
    _, index, _, _ = setup
    assert index.participants == ["alice", "jake"]
    assert len(index.episodes) == 18
    ep = index.episodes["A1_JAKE_DAY1_11000000_E01"]
    assert ep.location == "table" and "cell phone" in ep.objects
    assert index.stats["evidence_ids_missing"] == 0
    assert index.episode_of_row("A1_JAKE_DAY1_14000000#/3") == "A1_JAKE_DAY1_14000000_E04"


def test_index_normalisation_save_load_and_report(setup, tmp_path):
    from cemr.index_report import build_report
    _, index, _, _ = setup
    e03 = index.episodes["A1_JAKE_DAY1_14000000_E03"]          # Jake not listed in raw event
    assert "jake" in e03.people and "narrator_added" in e03.flags
    index.save(tmp_path / "idx")
    loaded = MemoryIndex.load(tmp_path / "idx")
    assert loaded.episodes.keys() == index.episodes.keys()
    assert len(loaded.rows) == len(index.rows)
    assert loaded.episode_rows("A1_JAKE_DAY1_11000000_E01") == index.episode_rows("A1_JAKE_DAY1_11000000_E01")
    rep = build_report(index)
    assert rep["summary"]["episodes"] == 18 and rep["readiness"]["T4_experienced_vs_told"]["ready"]


def test_all_miners_fire_and_respect_splits(setup):
    _, index, pipe, _ = setup
    res = pipe.mine()
    assert set(res) == {"T1", "T2", "T3", "T4", "T5"}
    for task, cands in res.items():
        assert cands, f"{task} produced no candidates"
        for c in cands:
            eps = [c.target_episode, *c.lure_episodes]
            days = {index.episodes[e].day for e in eps if e in index.episodes}
            assert len(days) <= 2 if task == "T3" else len(days) == 1, (task, c.candidate_id)
    subtypes = {c.subtype for c in res["T2"]}
    assert "executed_by_other" in subtypes
    assert any(c.lure_rows for c in res["T4"])


def test_end_to_end_generation_and_eval(setup):
    cfg, _, pipe, _ = setup
    pipe.mine()
    funnel = pipe.generate(limit_per_task=6)
    produced = sum(v.get("conv", 0) for v in funnel.values())
    assert produced > 0, json.dumps({k: dict(v) for k, v in funnel.items()})

    convs = list(read_jsonl(f"{cfg.output_dir}/conv.jsonl"))
    for c in convs:
        assert c["converged"]
        assert c["final_belief"] == c["gold_belief"]
        # every correcting assistant turn carries evidence
        for t in c["turns"]:
            if t["role"] == "assistant" and t["corrected_attribute"]:
                assert t["evidence_ids"]
        # primary error gets at least one push-back before acceptance
        if c["belief_condition"] == "misbound":
            assert c["flags"]["repeated_steps"] >= 1

    sft = list(read_jsonl(f"{cfg.output_dir}/sft/train.jsonl"))
    if sft:
        assert json.loads(sft[0]["messages"][2]["content"])["intent"]

    report = evaluate(pipe, pipe.ctx.generator, "mock-model", split="train", limit=3)
    assert "overall" in report and report["overall"]["n"] > 0


def _gt_fm():
    gt = GroundTruth(sample_id="s", task=TaskId.T1, narrator="Jake", target_episode="e",
                     facts=[Fact(fact_id="f1", slot="objects", value="cell phone", statement="x", evidence_ids=["r1"]),
                            Fact(fact_id="f2", slot="people", value="shure", statement="x", evidence_ids=["r2"])])
    fm = FalseMemory(sample_id="s", belief_condition=BeliefCondition.MISBOUND, user_memory="hard drive",
                     user_question="?", edits=[Edit(slot="objects", true="cell phone", false="hard drive",
                                                    error_type="event_blending", role="primary",
                                                    lure_source="e2", true_evidence=["r1"])])
    return gt, fm


def test_controller_rules():
    gt, fm = _gt_fm()
    eq = lambda s, a, b: a == b  # noqa: E731
    c = BeliefController(gt, fm, primary_resistance=1, secondary_resistance=0)
    # no evidence -> resisted, resistance NOT consumed
    assert c.react([Claim("objects", "cell phone", [])], "correct", eq, True).kind == "resist"
    # valid evidence: first push-back, then accept
    assert c.react([Claim("objects", "cell phone", ["r1"])], "correct", eq, True).kind == "resist"
    assert c.react([Claim("objects", "cell phone", ["r1"])], "correct", eq, True).kind == "accept"
    assert c.converged
    # correcting a true belief is logged as over-correction
    r = c.react([Claim("people", "lucia", ["r2"])], "correct", eq, True)
    assert r.invalid[0]["reason"] == "over_correction"


def test_leak_check():
    assert contains_value("I think it was the Hard Drive.", "hard drive")
    assert not contains_value("the harder drive", "hard drive")


###############################################################################
# Multi-participant behaviour
###############################################################################


def test_memory_bank_is_ego_centric(setup):
    _, index, pipe, _ = setup
    r = pipe.ctx.retriever_until(3 * 86400, "jake")
    assert r.episodes and all(e.participant == "jake" for e in r.episodes)
    assert all(e.participant == "alice" for e in index.bank_until(3 * 86400, "alice"))


def test_cross_view_clusters(setup):
    _, index, pipe, _ = setup
    j = index.episodes["A1_JAKE_DAY1_11000000_E01"]
    a = index.episodes["A2_ALICE_DAY1_11000000_E01"]
    assert j.cluster_id == a.cluster_id                       # same event, two wearers
    assert pipe.ctx.xv.n_views(j.episode_id) == 2
    # Alice's decorations episode does not list Jake -> not linked to his view
    assert index.episodes["A2_ALICE_DAY1_14000000_E01"].cluster_id != \
        index.episodes["A1_JAKE_DAY1_14000000_E03"].cluster_id


def test_miners_per_participant_and_verification(setup):
    _, index, pipe, _ = setup
    res = pipe.mine()
    for task, cands in res.items():
        for c in cands:
            assert index.episodes[c.target_episode].participant == c.participant
            for lure in c.lure_episodes:                          # lures from own memory
                assert index.episodes[lure].participant == c.participant
            for rid in c.lure_rows:                               # T4: only what the user heard
                assert index.rows[rid].participant == c.participant
    # T2: "Alice did it" is verified from Alice's OWN first-person caption
    other = [c for c in res["T2"] if c.subtype == "executed_by_other" and c.participant == "jake"]
    assert other and other[0].meta["actor_verified"] and "alice" in other[0].true_value
    assert all(index.rows[r].participant == "alice" for r in other[0].meta["verify_rows"])
    # near-duplicate cap per event cluster
    from collections import Counter
    per = Counter((c.task, index.episodes[c.target_episode].cluster_id) for cs in res.values() for c in cs)
    assert max(per.values()) <= pipe.cfg.mining.max_per_cluster


def test_heldout_participant_split(tmp_path):
    root = build_toy_dataset(tmp_path / "data")
    cfg = Config()
    cfg.data.captions_glob = str(root / "captions/*.json")
    cfg.data.events_glob = str(root / "events/*.json")
    cfg.data.train_days, cfg.data.dev_days, cfg.data.test_days = (1,), (2,), (3,)
    cfg.data.heldout_participants = ("alice",)
    cfg.data.index_dir = str(tmp_path / "idx")
    cfg.output_dir = str(tmp_path / "out")
    index = build_index(cfg)
    mock = MockLLM(index)
    pipe = Pipeline(cfg, index, {"generator": mock, "verifier": mock, "user_sim": mock})
    for cands in pipe.mine().values():
        for c in cands:
            if c.participant == "alice":
                assert c.split == "test_heldout_person" and index.episodes[c.target_episode].day == 3
            else:
                assert c.split in ("train", "dev", "test")


def test_visibility_is_deterministic(setup):
    from cemr.crossview import mark_visibility
    _, index, _, _ = setup
    facts = [Fact(fact_id="a", slot="objects", value="decorations", statement="x",
                  evidence_ids=["A2_ALICE_DAY1_14000000#/0"]),
             Fact(fact_id="b", slot="objects", value="tripod", statement="x",
                  evidence_ids=["A1_JAKE_DAY1_14000000#/4"])]
    mark_visibility(facts, "jake", index)
    assert facts[0].visible_to_user is False and facts[1].visible_to_user is True


def test_index_cleans_text_and_flags_bad_hour_files(tmp_path):
    """[cite: N] artefacts are stripped; copied caption tracks and caption/speech
    mismatches (both seen in the real Day-1 files) are flagged by the report."""
    from cemr.index_report import build_report

    def cap(t, text):
        return {"start": f"00:{t // 60:02d}:{t % 60:02d},000", "end": f"00:{t // 60:02d}:{t % 60:02d},900",
                "kind": "caption", "text_en": text, "speaker": None}

    def utt(t, text):
        return {**cap(t, text), "kind": "utterance", "speaker": "Jake"}

    (tmp_path / "captions").mkdir()
    hour14 = [cap(t, f"I wash the dishes[cite: 4] {t}.") for t in range(0, 1200, 30)] + \
             [utt(t, "Very Chinese.") for t in range(0, 1200, 60)]
    # 17:00 = the 14:00 caption track (same timestamps, re-paraphrased) + speech from later in the hour
    hour17 = [cap(t, f"Washing dishes {t}.") for t in range(0, 1200, 30)] + \
             [utt(t, "Fresh tofu.") for t in range(2400, 3600, 60)]
    for hh, rows in (("14", hour14), ("17", hour17)):
        (tmp_path / "captions" / f"A1_JAKE_DAY1_{hh}000000.json").write_text(json.dumps(rows))
    idx = MemoryIndex.build(str(tmp_path / "captions" / "*.json"), str(tmp_path / "none" / "*.json"))
    assert all("[cite" not in r.text for r in idx.rows.values())
    rep = build_report(idx)["integrity"]
    assert rep["duplicate_caption_tracks"] and rep["duplicate_caption_tracks"][0][:2] == [
        "A1_JAKE_DAY1_14000000", "A1_JAKE_DAY1_17000000"]
    assert rep["caption_utterance_mismatch"] == ["A1_JAKE_DAY1_17000000"]


def test_evidence_ids_one_based_case_and_recovery(tmp_path):
    """Real EgoLife events: 1-based row numbers, odd casing, ids missing hour and '#/'."""
    (tmp_path / "captions").mkdir(); (tmp_path / "events").mkdir()
    rows = [{"start": f"00:00:0{i},000", "end": f"00:00:0{i},900", "kind": "caption",
             "text_en": f"row {i}", "speaker": None} for i in range(5)]
    (tmp_path / "captions" / "A4_LUCIA_DAY1_12000000.json").write_text(json.dumps(rows))
    ev = {"event_id": "E1", "start_time": "00:00:00.000", "end_time": "00:00:04.900",
          "people": ["Lucia", "dog owner,Alice / Tasha"],
          "evidence_ids": ["A4_LucIA_DAY1_12000000#/1", "A4_LUCIA_DAY1_12000000#/5",
                           "A4_LUCIA_DAY1_3", "A6_SHURE_DAY4_9", "junk"]}
    (tmp_path / "events" / "A4_LUCIA_DAY1_12000000.json").write_text(json.dumps({"events": [ev]}))
    idx = MemoryIndex.build(str(tmp_path / "captions/*.json"), str(tmp_path / "events/*.json"),
                            people=("lucia", "alice", "tasha"), evidence_index_base=1)
    ep = idx.episodes["E1"]
    assert ep.evidence_ids == ["A4_LUCIA_DAY1_12000000#/0", "A4_LUCIA_DAY1_12000000#/4",
                               "A4_LUCIA_DAY1_12000000#/2"]
    assert all(x in idx.rows for x in ep.evidence_ids) and "dangling_evidence" not in ep.flags
    assert idx.stats["evidence_ids_recovered"] == 1 and idx.stats["evidence_ids_malformed"] == 2
    assert {"alice", "tasha", "dog owner", "lucia"} <= set(ep.people)


def test_bad_event_times_are_skipped_not_fatal(tmp_path):
    (tmp_path / "captions").mkdir(); (tmp_path / "events").mkdir()
    rows = [{"start": "00:00:01,000", "end": "00:00:02,000", "kind": "caption", "text_en": "x", "speaker": None}]
    (tmp_path / "captions" / "A1_JAKE_DAY1_11000000.json").write_text(json.dumps(rows))
    evs = [{"event_id": "ok4", "start_time": "00:00:01:500", "end_time": "00:00:02:000", "evidence_ids": []},
           {"event_id": "bad", "start_time": "00:00:01:2:3", "end_time": "00:00:02", "evidence_ids": []},
           {"event_id": "rev", "start_time": "00:08:12,333", "end_time": "00:05:00,000", "evidence_ids": []}]
    (tmp_path / "events" / "A1_JAKE_DAY1_11000000.json").write_text(json.dumps({"events": evs}))
    idx = MemoryIndex.build(str(tmp_path / "captions/*.json"), str(tmp_path / "events/*.json"))
    assert set(idx.episodes) == {"ok4"} and idx.stats["events_bad_time"] == 2
