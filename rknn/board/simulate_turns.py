#!/usr/bin/env python3
"""Compare end-of-turn policies built from the firmware's VAD and Smart Turn.

Two steps, so a whole grid of policy settings comes from one pass on the board:

  record  (on the board) Feed each recording in 32 ms frames, followed by
          --tail-ms of silence, to the firmware's Silero VAD helper
          (rknn_vad). Whenever a pause inside the turn reaches one of the
          --trigger-ms lengths, ask Smart Turn (smart_turn_helper) about the
          audio so far. Writes one JSON line per recording: the VAD
          probability of every frame, and Smart Turn's probability and call
          time at each of those points.
  sweep   (anywhere) Replay the recorded decisions for a grid of policies and
          print a comparison table.

Policies:
  silence:<ms>                 end the turn after <ms> of VAD silence (the
                               firmware's current rule: silence_ms = 550)
  smart:<trigger>/<fallback>/<threshold>
                               after <trigger> ms of silence, ask Smart Turn;
                               end the turn if its probability > <threshold>,
                               otherwise keep waiting, asking again whenever a
                               new pause reaches <trigger>, and end it at
                               <fallback> ms of silence. pipecat's default is
                               200 / 3000 / 0.5.

The recording's speech end is the end of its last VAD speech frame. A policy
"cuts" a recording if it ends the turn before that point; latency is measured
from that point. Recordings labelled incomplete stop mid-thought, so a policy
should keep those turns open.

Pure Python (no numpy), so `record` runs on the device.
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import time

RATE = 16000
FRAME = 512  # Silero VAD frame: 32 ms
FRAME_MS = FRAME * 1000 / RATE


def frames_for(ms):
    """Frames of silence needed to reach `ms` (the VAD counts whole frames)."""
    return math.ceil(ms / FRAME_MS - 1e-9)


class TurnState:
    """VAD bookkeeping shared by `record` and `sweep`, as the firmware's VAD loop counts it."""

    def __init__(self, threshold, min_speech_ms):
        self.threshold, self.min_speech_ms = threshold, min_speech_ms
        self.speech_ms, self.silence = 0.0, 0

    def update(self, prob):
        """Returns (is_speech, silence_frames, turn_started)."""
        is_speech = prob > self.threshold
        if is_speech:
            self.speech_ms += FRAME_MS
            self.silence = 0
        elif self.speech_ms >= self.min_speech_ms:
            self.silence += 1
        return is_speech, self.silence, self.speech_ms >= self.min_speech_ms


# --------------------------------------------------------------------------
# record (on the board)

class Helper:
    """A line-oriented helper process (READY on startup, one reply line per request)."""

    def __init__(self, args):
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        line = self.proc.stdout.readline().decode().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"{args[0]} did not start: {line!r}")
        self.ready = line

    def send(self, payload):
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        return self.proc.stdout.readline().decode().strip()

    def request(self, payload):
        line = self.send(payload)
        if not line.startswith("P "):
            raise RuntimeError(f"helper error: {line!r}")
        return line[2:].split()

    def close(self):
        try:
            self.proc.stdin.write(b"Q")
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait(timeout=5)


def read_records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            (n,) = struct.unpack("<I", head)
            yield f.read(2 * n)


def record_one(pcm, vad, turn, window_bytes, triggers, tail_frames, args):
    n_frames = len(pcm) // (2 * FRAME)
    pcm = pcm[: n_frames * 2 * FRAME] + bytes(2 * FRAME * tail_frames)
    vad.send(b"R")
    state = TurnState(args.vad_threshold, args.min_speech_ms)
    probs, calls = [], {}
    start = time.monotonic()
    for f in range(n_frames + tail_frames):
        if args.realtime:
            delay = start + (f + 1) * FRAME_MS / 1000 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        upto = (f + 1) * 2 * FRAME
        # Round before use, so `sweep` sees exactly the values the state used.
        prob = round(float(vad.request(b"F" + pcm[upto - 2 * FRAME:upto])[0]), 4)
        probs.append(prob)
        is_speech, silence, started = state.update(prob)
        if started and not is_speech and silence in triggers:
            audio = pcm[max(0, upto - window_bytes):upto]
            t0 = time.monotonic()
            p, features_ms, run_ms = (float(v) for v in turn.request(
                b"T" + struct.pack("<I", len(audio) // 2) + audio))
            wall = (time.monotonic() - t0) * 1000
            calls[str(f)] = [round(p, 4), round(wall, 1), features_ms, run_ms]
    return {"audio_frames": n_frames, "vad": probs, "calls": calls}


def cmd_record(args):
    labels = [int(line) for line in open(args.labels) if line.strip()]
    vad = Helper([args.vad_helper, "--model", args.vad_model, "--weights", args.vad_weights])
    turn = Helper([args.smart_turn_helper, "--model", args.smart_turn_model])
    window_bytes = 2 * int(turn.ready.split()[1])
    triggers = {frames_for(ms) for ms in args.trigger_ms}
    tail_frames = frames_for(args.tail_ms)
    meta = {"vad_threshold": args.vad_threshold, "min_speech_ms": args.min_speech_ms,
            "trigger_frames": sorted(triggers), "tail_frames": tail_frames,
            "smart_turn_model": args.smart_turn_model, "realtime": args.realtime}
    done = 0
    if args.resume:
        # Continue an interrupted run: keep the complete lines, append the rest.
        with open(args.out) as f:
            if json.loads(f.readline())["meta"] != meta:
                sys.exit("--resume: settings differ from the existing output")
            lines = f.readlines()
        good = []
        for line in lines:
            try:
                json.loads(line)
            except ValueError:
                break  # a line cut off by the interruption
            good.append(line)
        done = len(good)
        with open(args.out, "w") as f:
            f.write(json.dumps({"meta": meta}) + "\n")
            f.writelines(good)
        print(f"resuming after {done} recordings", file=sys.stderr, flush=True)
    with open(args.out, "a" if args.resume else "w") as out:
        if not args.resume:
            out.write(json.dumps({"meta": meta}) + "\n")
        for i, pcm in enumerate(read_records(args.audio)):
            if args.count and i >= args.count:
                break
            if i < done:
                continue
            rec = record_one(pcm, vad, turn, window_bytes, triggers, tail_frames, args)
            out.write(json.dumps({"label": labels[i], **rec}) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1} recordings", file=sys.stderr, flush=True)
    vad.close()
    turn.close()


# --------------------------------------------------------------------------
# sweep (anywhere)

def run_policy(rec, meta, policy):
    """Return (end_ms, reason) for one recording, or (None, None) if no turn ended."""
    kind, params = policy
    state = TurnState(meta["vad_threshold"], meta["min_speech_ms"])
    for f, prob in enumerate(rec["vad"]):
        is_speech, silence, started = state.update(prob)
        if not started or is_speech:
            continue
        now = (f + 1) * FRAME_MS
        if kind == "silence":
            if silence >= params:
                return now, "silence"
            continue
        trigger, fallback, threshold = params
        if silence == trigger:
            p, wall = rec["calls"][str(f)][:2]
            if p > threshold:
                return now + wall, "smart"  # the decision exists once inference returns
        if silence >= fallback:
            return now, "fallback"
    return None, None


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def rounded(value):
    return None if value is None or math.isnan(value) else round(value)


def evaluate(recs, meta, policy, keep_open_ms):
    cut, latency, from_fallback, open_ok, n_incomplete, no_end = 0, [], 0, 0, 0, 0
    for rec in recs:
        speech = [f for f, p in enumerate(rec["vad"]) if p > meta["vad_threshold"]]
        end, reason = run_policy(rec, meta, policy)
        if end is None or not speech:
            no_end += 1
            continue
        speech_end = (speech[-1] + 1) * FRAME_MS
        if end < speech_end:
            cut += 1
        elif rec["label"]:
            latency.append(end - speech_end)
            from_fallback += reason == "fallback"
        else:
            n_incomplete += 1
            open_ok += end - speech_end >= keep_open_ms
    return {
        "cut": cut,
        "complete": len(latency),
        "latency_median": rounded(percentile(latency, 0.5)),
        "latency_p90": rounded(percentile(latency, 0.9)),
        "latency_mean": rounded(sum(latency) / len(latency)) if latency else None,
        "complete_waited_fallback": from_fallback,
        "incomplete": n_incomplete,
        "incomplete_kept_open": open_ok,
        "no_turn": no_end,
    }


def parse_policy(text):
    kind, _, rest = text.partition(":")
    if kind == "silence":
        return text, ("silence", frames_for(float(rest)))
    trigger, fallback, threshold = rest.split("/")
    return text, ("smart", (frames_for(float(trigger)), frames_for(float(fallback)), float(threshold)))


def cmd_sweep(args):
    with open(args.records) as f:
        meta = json.loads(f.readline())["meta"]
        recs = [json.loads(line) for line in f]
    policies = [f"silence:{ms:g}" for ms in args.silence_ms]
    policies += [f"smart:{t:g}/{fb:g}/{th:g}" for t in args.trigger_ms for fb in args.fallback_ms
                 for th in args.threshold]
    for name in policies:
        kind, params = parse_policy(name)[1]
        if kind == "smart":
            if params[0] not in meta["trigger_frames"]:
                sys.exit(f"{name}: trigger was not recorded (recorded frames: {meta['trigger_frames']})")
            if params[1] > meta["tail_frames"]:
                sys.exit(f"{name}: fallback is longer than the recorded silence tail")

    calls = [c for r in recs for c in r["calls"].values()]
    print(f"{len(recs)} recordings ({sum(r['label'] for r in recs)} complete), "
          f"{len(calls)} Smart Turn calls: wall median {percentile([c[1] for c in calls], 0.5):.1f} ms, "
          f"p90 {percentile([c[1] for c in calls], 0.9):.1f} ms")
    print(f"| policy | cut before speech end | complete: latency median / p90 / mean (ms) | "
          f"complete waited for fallback | incomplete kept open >= {args.keep_open_ms:g} ms |")
    print("|---|---|---|---|---|")
    results = {}
    for name in policies:
        r = results[name] = evaluate(recs, meta, parse_policy(name)[1], args.keep_open_ms)
        print(f"| {name} | {r['cut']} | {r['latency_median']} / {r['latency_p90']} / {r['latency_mean']} | "
              f"{r['complete_waited_fallback']} / {r['complete']} | "
              f"{r['incomplete_kept_open']} / {r['incomplete']} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="run VAD and Smart Turn on the board")
    p.add_argument("--audio", required=True, help="records of [u32 n][int16 n], 16 kHz mono")
    p.add_argument("--labels", required=True, help="one 0/1 per line (1 = ends at a complete turn)")
    p.add_argument("--out", required=True, help="JSON lines output")
    p.add_argument("--count", type=int, default=0, help="only the first N recordings")
    p.add_argument("--smart-turn-helper", required=True)
    p.add_argument("--smart-turn-model", required=True)
    p.add_argument("--vad-helper", default="/usr/lib/aiden/rknn_vad")
    p.add_argument("--vad-model", default="/usr/lib/aiden/models/silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn")
    p.add_argument("--vad-weights", default="/usr/lib/aiden/models/silero_vad_6_2_lstm_decoder_weights.bin")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--min-speech-ms", type=float, default=300)
    p.add_argument("--trigger-ms", type=float, nargs="+", default=[200, 300, 400, 500],
                   help="pause lengths at which to ask Smart Turn")
    p.add_argument("--tail-ms", type=float, default=3500, help="silence appended to each recording")
    p.add_argument("--realtime", action="store_true", help="pace frames at 32 ms like a live microphone")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run in --out")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("sweep", help="compare policies on recorded decisions")
    p.add_argument("--records", required=True, help="output of `record`")
    p.add_argument("--silence-ms", type=float, nargs="*", default=[550, 800, 1000])
    p.add_argument("--trigger-ms", type=float, nargs="*", default=[200, 300, 400, 500])
    p.add_argument("--fallback-ms", type=float, nargs="*", default=[1500, 2000, 3000])
    p.add_argument("--threshold", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8])
    p.add_argument("--keep-open-ms", type=float, default=1000,
                   help="an incomplete turn counts as kept open if it lasts this long after the speech")
    p.add_argument("--json", help="also write all results as JSON")
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
