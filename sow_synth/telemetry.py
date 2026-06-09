"""Run telemetry — step timing and LLM token tracking.

Usage in generate_sample.py:

    tel = Telemetry()
    with tel.step("stage_1_profile"):
        profile = resolve_profile(spec, rng)
    # LlmClient records its own calls directly via tel.record_llm_call(...)
    tel.print_report()
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    name: str
    duration_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def is_llm(self) -> bool:
        return self.total_tokens > 0


class Telemetry:
    def __init__(self) -> None:
        self._steps: list[StepRecord] = []
        self._run_start: float = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        """Context manager: record wall-clock time for a non-LLM stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._steps.append(StepRecord(name=name, duration_s=time.perf_counter() - t0))

    def record_llm_call(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
    ) -> None:
        """Record a completed LLM call with token usage."""
        self._steps.append(StepRecord(
            name=name,
            duration_s=duration_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    # ------------------------------------------------------------------

    def print_report(self) -> None:
        total_tokens = sum(s.total_tokens for s in self._steps)
        total_time   = time.perf_counter() - self._run_start

        W = 78
        col = dict(name=36, time=8, prompt=8, compl=8, total=8, share=7)

        def _row(name, time_s, prompt, compl, total, share, *, header=False, sep=False):
            if sep:
                print("-" * W)
                return
            ts  = f"{time_s:.3f}s" if isinstance(time_s, float) else time_s
            pt  = str(prompt) if prompt else "-"
            ct  = str(compl)  if compl  else "-"
            tt  = str(total)  if total  else "-"
            sh  = f"{share:.1f}%" if isinstance(share, float) else share
            print(f"  {name:<{col['name']}} {ts:>{col['time']}}  "
                  f"{pt:>{col['prompt']}}  {ct:>{col['compl']}}  "
                  f"{tt:>{col['total']}}  {sh:>{col['share']}}")

        SEP1 = "=" * W
        SEP2 = "-" * W
        print()
        print(SEP1)
        print("  TELEMETRY REPORT")
        print(SEP1)
        print(f"  {'Step':<{col['name']}} {'Time':>{col['time']}}  "
              f"{'Prompt':>{col['prompt']}}  {'Compl.':>{col['compl']}}  "
              f"{'Total':>{col['total']}}  {'Share':>{col['share']}}")
        print(SEP2)

        for s in self._steps:
            share = (s.total_tokens / total_tokens * 100) if total_tokens and s.is_llm else ""
            _row(
                s.name, s.duration_s,
                s.prompt_tokens if s.is_llm else 0,
                s.completion_tokens if s.is_llm else 0,
                s.total_tokens if s.is_llm else 0,
                share,
            )

        print(SEP2)

        total_prompt = sum(s.prompt_tokens for s in self._steps)
        total_compl  = sum(s.completion_tokens for s in self._steps)
        print(f"  {'TOTAL':<{col['name']}} {total_time:>{col['time']}.3f}s  "
              f"{total_prompt:>{col['prompt']}}  {total_compl:>{col['compl']}}  "
              f"{total_tokens:>{col['total']}}  {'100.0%':>{col['share']}}")
        print(SEP1)
        print()
