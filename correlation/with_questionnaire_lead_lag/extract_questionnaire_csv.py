"""
Questionnaire extractor — no windowing; shared across all lag levels.

Extracts all questionnaire responses with their timestamps and
initiation times.  The output is used by merge_multimodal_csv.py to
attach questionnaire columns (effort, frustration, mental_demand, etc.)
to the multimodal merged CSV.

Output: processed_data/questionnaire/42-questionnaires.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_definition import psydat_files
from data_parser import DataParser


class PerformanceAnswerExtractor:
    name = "performance"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"]   for e in questionnaires]
        y = [e[f"{q}.rating"]                                for e in questionnaires]
        t = [e["browser_content.started"]                    for e in questionnaires]
        return x, y, t


class TemporalDemandAnswerExtractor:
    name = "temporal_demand"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


class AttentivenessAnswerExtractor:
    name = "attentiveness"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "attentiveness:_how_focused_were_you_on_performing_the_task_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


class SleepinessAnswerExtractor:
    name = "sleepiness"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "sleepiness:_how_sleepy_are_you_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


class EffortAnswerExtractor:
    name = "effort"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


class FrustrationAnswerExtractor:
    name = "frustration"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


class MentalDemandAnswerExtractor:
    name = "mental_demand"

    def process(self, parser):
        questionnaires = parser["browser_content"]
        q = "mental_demand:_how_mentally_demanding_was_the_task_slider"
        x = [e["browser_content.started"] + e[f"{q}.rt"] for e in questionnaires]
        y = [e[f"{q}.rating"]                              for e in questionnaires]
        t = [e["browser_content.started"]                  for e in questionnaires]
        return x, y, t


_EXTRACTORS = [
    PerformanceAnswerExtractor(),
    TemporalDemandAnswerExtractor(),
    AttentivenessAnswerExtractor(),
    SleepinessAnswerExtractor(),
    EffortAnswerExtractor(),
    FrustrationAnswerExtractor(),
    MentalDemandAnswerExtractor(),
]


def run(output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.path.join("processed_data", "questionnaire")

    processed: dict = {}
    for psydat_file in tqdm(psydat_files, desc="questionnaire"):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        for extractor in _EXTRACTORS:
            times, values, initiations = extractor.process(parser)
            processed.setdefault("name",        []).extend([extractor.name] * len(values))
            processed.setdefault("time",        []).extend(times)
            processed.setdefault("value",       []).extend(values)
            processed.setdefault("initiation",  []).extend(initiations)
            processed.setdefault("participant", []).extend([participant_id] * len(values))

    df = pd.DataFrame(processed)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{len(psydat_files)}-questionnaires.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()
    run(output_dir=args.output_dir)
