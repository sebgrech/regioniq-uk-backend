#!/usr/bin/env python3
"""
Region IQ - Full Forecast Pipeline Orchestrator (V2.3)
======================================================

Runs complete RegionIQ data + forecast cascade with vintage snapshots:

  Pre-Snapshot → Ingest → Transform → Macro FCST → Macro QA →
  ITL1 FCST → ITL1 QA →
  ITL2 FCST → ITL2 QA →
  ITL3 FCST → ITL3 QA →
  LAD FCST → LAD QA →
  Supabase Sync → Post-Snapshot → Email Report

Design principles:
- Fail-fast: stop immediately on any critical QA failure
- Fully modular stages with resume capability
- All logs persisted under /data/logs/pipeline_<run_id>
- 100% deterministic execution order
- Vintage snapshots track table changes across runs
- Email report sent on successful completion

Usage:
    python3 run_full_forecast_pipeline.py
    python3 run_full_forecast_pipeline.py --start-from macro
    python3 run_full_forecast_pipeline.py --start-from itl3 --stop-at lad
    python3 run_full_forecast_pipeline.py --stop-at supabase  # skip email

Stages (in order):
    pre_snapshot → ingest → transform → macro → itl1 → itl2 → itl3 → lad → supabase → post_snapshot → email_report

Author: Region IQ
Version: 2.3 (Vintage snapshots + email notification)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List
from dataclasses import dataclass, asdict
import time

# -------------------------------------------------------------------
# ANSI COLORS
# -------------------------------------------------------------------

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


# -------------------------------------------------------------------
# STAGE RESULT DATACLASS
# -------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    stage_type: str  # ingest / transform / forecast / qa / sync / snapshot
    status: str      # success / failed / skipped
    duration_seconds: float
    exit_code: int
    error_message: Optional[str] = None
    warnings: int = 0
    critical_issues: int = 0


# -------------------------------------------------------------------
# MAIN ORCHESTRATOR CLASS
# -------------------------------------------------------------------

class ForecastPipeline:
    """Orchestrates full multi-level RegionIQ forecast cascade."""

    # FULL PIPELINE CONFIG
    STAGES = {
        'pre_snapshot': {
            'scripts': ['scripts/manifest/pre_vintage_snapshot.py'],
            'qa': None,
            'pass_run_id': True
        },
        'ingest': {
            'scripts': [
                'scripts/ingest/BROAD_Macro_T.py',
                'scripts/ingest/Broad_ITL1_T.py',
                'scripts/ingest/Broad_LAD_T.py'
            ],
            'qa': None,
            'pass_run_id': False
        },
        'transform': {
            'scripts': ['scripts/transform/Broad_transform.py'],
            'qa': None,
            'pass_run_id': False
        },
        'fill_lad_gaps': {
            # fill_lad_rate_gaps.py updates silver.lad_history and also re-aggregates
            # filled LAD rates back to silver.itl3_history / silver.itl2_history.
            # Do NOT rerun Broad_transform here, because it rebuilds ITL histories
            # from the *raw* LAD rate history tables (without interpolated values),
            # which would overwrite the gap-filled series.
            'scripts': ['scripts/transform/fill_lad_rate_gaps.py'],
            'qa': None,
            'pass_run_id': False
        },
        'macro': {
            'scripts': ['scripts/forecast/Broad_Macro_forecast.py'],
            'qa': 'scripts/forecast/QA/Macro_Broadbased_QA.py',
            'pass_run_id': False
        },
        'itl1': {
            'scripts': ['scripts/forecast/Broad_ITL1_forecast.py'],
            'qa': 'scripts/forecast/QA/ITL1_Broadbased_QA.py',
            'pass_run_id': False
        },
        'itl2': {
            'scripts': ['scripts/forecast/Broad_ITL2_forecast.py'],
            'qa': 'scripts/forecast/QA/ITL2_Broadbased_QA.py',
            'pass_run_id': False
        },
        'itl3': {
            'scripts': ['scripts/forecast/Broad_ITL3_forecast.py'],
            'qa': 'scripts/forecast/QA/ITL3_Broadbased_QA.py',
            'pass_run_id': False
        },
        'lad': {
            'scripts': [
                'scripts/forecast/Broad_LAD_forecast.py'
            ],
            'qa': 'scripts/forecast/QA/LAD_Broadbased_QA.py',
            'pass_run_id': False
        },
        'supabase': {
            'scripts': ['scripts/export/Broad_export.py'],
            'qa': None,
            'pass_run_id': False
        },
        'post_snapshot': {
            'scripts': ['scripts/manifest/post_vintage_snapshot.py'],
            'qa': None,
            'pass_run_id': True
        },
        'email_report': {
            'scripts': ['scripts/notify/weekly_email.py'],
            'qa': None,
            'pass_run_id': True
        }
    }

    STAGE_ORDER = [
        'pre_snapshot',
        # IMPORTANT: fill LAD rate gaps BEFORE aggregating to ITL levels (transform),
        # so ITL3/ITL2/ITL1 rates inherit gap-filled LAD histories.
        'ingest', 'fill_lad_gaps', 'transform',
        'macro', 'itl1', 'itl2', 'itl3',
        'lad',
        'supabase',
        'post_snapshot',
        'email_report'
    ]

    # -------------------------------------------------------------------

    def __init__(self, start_from=None, stop_at=None):
        self.start_from = start_from or 'pre_snapshot'
        self.stop_at = stop_at or 'email_report'
        self.results: List[StageResult] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(f"data/logs/pipeline_{self.run_id}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._validate_stages()

    # -------------------------------------------------------------------

    def _validate_stages(self):
        """Ensure valid start and stop ranges."""
        if self.start_from not in self.STAGE_ORDER:
            raise ValueError(f"Invalid start stage: {self.start_from}")

        if self.stop_at not in self.STAGE_ORDER:
            raise ValueError(f"Invalid stop stage: {self.stop_at}")

        if self.STAGE_ORDER.index(self.start_from) > self.STAGE_ORDER.index(self.stop_at):
            raise ValueError(f"start_from '{self.start_from}' occurs AFTER stop_at '{self.stop_at}'")

    # -------------------------------------------------------------------

    def _print_header(self):
        print(f"\n{Colors.HEADER}{'=' * 90}{Colors.END}")
        print(f"{Colors.HEADER}{Colors.BOLD}REGION IQ — FULL FORECAST PIPELINE (V2.3 WITH EMAIL NOTIFICATION){Colors.END}")
        print(f"{Colors.HEADER}{'=' * 90}{Colors.END}")
        print(f"{Colors.CYAN}Run ID: {self.run_id}{Colors.END}")

        start_idx = self.STAGE_ORDER.index(self.start_from)
        stop_idx = self.STAGE_ORDER.index(self.stop_at)
        stages = " → ".join(s.upper() for s in self.STAGE_ORDER[start_idx:stop_idx+1])

        print(f"{Colors.CYAN}Execution Chain: {stages}{Colors.END}")
        print(f"{Colors.CYAN}Logs stored at: {self.log_dir}{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 90}{Colors.END}\n")

    # -------------------------------------------------------------------

    def _print_stage_header(self, name, stage_type):
        icon = {
            'forecast': "📈",
            'qa': "🔍",
            'ingest': "📥",
            'transform': "🧪",
            'sync': "🔄",
            'snapshot': "📸",
            'notify': "📧"
        }.get(stage_type, "📦")

        print(f"\n{Colors.BLUE}{Colors.BOLD}{icon} {name.upper()} — {stage_type.upper()}{Colors.END}")
        print(f"{Colors.BLUE}{'-' * 60}{Colors.END}")

    # -------------------------------------------------------------------

    def _get_script_label(self, script: str, stage: str) -> str:
        """Extract a readable label from script path for multi-script stages."""
        filename = Path(script).stem.lower()
        
        # Map known patterns to clean labels
        if 'macro' in filename:
            return f"{stage}_macro"
        elif 'itl1' in filename or 'itl_1' in filename:
            return f"{stage}_itl1"
        elif 'itl2' in filename or 'itl_2' in filename:
            return f"{stage}_itl2"
        elif 'itl3' in filename or 'itl_3' in filename:
            return f"{stage}_itl3"
        elif 'lad' in filename or 'la_' in filename or '_la' in filename:
            return f"{stage}_lad"
        else:
            return stage

    def _run_command(self, script: str, stage: str, stage_type: str, pass_run_id: bool = False) -> StageResult:
        """Execute a single script (forecast, QA, or snapshot)."""
        
        # Get descriptive label for multi-script stages
        label = self._get_script_label(script, stage)

        self._print_stage_header(label, stage_type)

        path = Path(script)
        if not path.exists():
            return StageResult(
                name=f"{label}_{stage_type}",
                stage_type=stage_type,
                status="failed",
                duration_seconds=0,
                exit_code=1,
                error_message=f"Script not found: {script}"
            )

        log_file = self.log_dir / f"{label}_{stage_type}.log"

        # Build command with optional run_id
        cmd = ["python3", str(path)]
        if pass_run_id:
            cmd.extend(["--run-id", self.run_id])

        start = time.time()
        try:
            with open(log_file, "w") as f:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600
                )
                f.write(process.stdout)
                print(process.stdout)

            duration = time.time() - start
            exit_code = process.returncode

            warnings = self._parse_qa_warnings(process.stdout)
            critical = self._parse_qa_critical(process.stdout) if exit_code != 0 else 0

            return StageResult(
                name=f"{label}_{stage_type}",
                stage_type=stage_type,
                status="success" if exit_code == 0 else "failed",
                duration_seconds=duration,
                exit_code=exit_code,
                warnings=warnings,
                critical_issues=critical
            )

        except subprocess.TimeoutExpired:
            return StageResult(
                name=f"{label}_{stage_type}",
                stage_type=stage_type,
                status="failed",
                duration_seconds=time.time() - start,
                exit_code=124,
                error_message="Timeout expired (3600s)"
            )

    # -------------------------------------------------------------------

    def _parse_qa_warnings(self, txt: str) -> int:
        if "Warnings:" in txt:
            try: return int(txt.split("Warnings:")[-1].strip().split()[0])
            except: pass
        return 0

    def _parse_qa_critical(self, txt: str) -> int:
        if "Critical Issues:" in txt:
            try: return int(txt.split("Critical Issues:")[-1].strip().split()[0])
            except: pass
        return 1

    # -------------------------------------------------------------------

    def _should_run_stage(self, stage):
        start = self.STAGE_ORDER.index(self.start_from)
        stop = self.STAGE_ORDER.index(self.stop_at)
        idx = self.STAGE_ORDER.index(stage)
        return start <= idx <= stop

    # -------------------------------------------------------------------

    def _get_stage_type(self, stage: str) -> str:
        """Determine stage type for display purposes."""
        if stage in ('pre_snapshot', 'post_snapshot'):
            return 'snapshot'
        elif stage == 'ingest':
            return 'ingest'
        elif stage == 'transform':
            return 'transform'
        elif stage == 'supabase':
            return 'sync'
        elif stage == 'email_report':
            return 'notify'
        else:
            return 'forecast'

    # -------------------------------------------------------------------

    def run(self) -> bool:
        """Run full multi-stage pipeline with QA gating."""

        self._print_header()
        pipeline_start = time.time()

        for stage in self.STAGE_ORDER:
            if not self._should_run_stage(stage):
                continue

            stage_config = self.STAGES[stage]
            stage_type = self._get_stage_type(stage)
            pass_run_id = stage_config.get('pass_run_id', False)

            scripts = stage_config['scripts']
            qa_script = stage_config['qa']

            # Run main scripts for this stage
            for script in scripts:
                res = self._run_command(script, stage, stage_type, pass_run_id=pass_run_id)
                self.results.append(res)

                if res.status == "failed":
                    self._print_failure_summary(res)
                    self._save_summary(False)
                    return False

            # Run QA gate if defined
            if qa_script:
                qa_res = self._run_command(qa_script, stage, "qa", pass_run_id=False)
                self.results.append(qa_res)

                if qa_res.status == "failed":
                    self._print_failure_summary(qa_res)
                    self._save_summary(False)
                    return False

            print(f"\n{Colors.GREEN}✔ {stage.upper()} COMPLETE{Colors.END}")

        total_time = time.time() - pipeline_start
        self._print_success_summary(total_time)
        self._save_summary(True)
        return True

    # -------------------------------------------------------------------

    def _print_failure_summary(self, result: StageResult):
        print(f"\n{Colors.RED}{'='*90}{Colors.END}")
        print(f"{Colors.RED}{Colors.BOLD}❌ PIPELINE HALTED — CRITICAL FAILURE{Colors.END}")
        print(f"{Colors.RED}{'='*90}{Colors.END}")
        print(f"{Colors.RED}Stage: {result.name}{Colors.END}")
        print(f"{Colors.RED}Exit: {result.exit_code}{Colors.END}")
        if result.error_message:
            print(f"{Colors.RED}Error: {result.error_message}{Colors.END}")

    # -------------------------------------------------------------------

    def _print_success_summary(self, total_seconds):
        print(f"\n{Colors.GREEN}{'='*90}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}PIPELINE COMPLETE — ALL STAGES PASSED{Colors.END}")
        print(f"{Colors.GREEN}{'='*90}{Colors.END}")

        print(f"\n{Colors.CYAN}Stage Results:{Colors.END}")
        for r in self.results:
            icon = "✔" if r.status == "success" else "✖"
            print(f"  {icon} {r.name:30s} {r.duration_seconds:6.1f}s")

        print(f"\n{Colors.CYAN}Total time: {total_seconds:.1f}s ({total_seconds/60:.1f} minutes){Colors.END}")

        print(f"\nLogs stored under: {self.log_dir}")

    # -------------------------------------------------------------------

    def _save_summary(self, success: bool):
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "stages": [asdict(r) for r in self.results]
        }

        out = self.log_dir / "pipeline_summary.json"
        json.dump(summary, open(out, "w"), indent=2)

        print(f"{Colors.CYAN}Summary JSON: {out}{Colors.END}")


# -------------------------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-from",
        choices=ForecastPipeline.STAGE_ORDER,
        default="pre_snapshot"
    )

    parser.add_argument(
        "--stop-at",
        choices=ForecastPipeline.STAGE_ORDER,
        default="post_snapshot"
    )

    args = parser.parse_args()

    try:
        pipeline = ForecastPipeline(
            start_from=args.start_from,
            stop_at=args.stop_at
        )
        ok = pipeline.run()
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrupted by user.{Colors.END}")
        sys.exit(130)


if __name__ == "__main__":
    main()