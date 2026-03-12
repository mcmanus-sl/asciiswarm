#!/usr/bin/env python3
"""Custom test runner with clean output for AGENT SWARM consumption.

Usage:
    python run_tests.py              # Full suite
    python run_tests.py --fast       # Deterministic 10% sample
    python run_tests.py --fast --seed 42
    python run_tests.py --log test_failures.log

Output format:
    PASS kernel/test_env.py (23 tests)
    FAIL kernel/test_events.py
      ERROR: test_handler_fires — assert 1 == 2
    TOTAL: 247/250 passed
"""

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

import pytest


class QuietCollector:
    """Pytest plugin that collects results per file with minimal output."""

    def __init__(self):
        self.file_results = {}  # filepath -> {'passed': int, 'failed': [], 'total': int}
        self.total_passed = 0
        self.total_failed = 0
        self.failure_details = []  # (filepath, test_name, short_msg, full_traceback)

    def pytest_runtest_logreport(self, report):
        if report.when != 'call':
            return

        # Get relative path from tests/
        fspath = report.fspath
        try:
            rel = os.path.relpath(fspath, 'tests')
        except ValueError:
            rel = fspath

        if rel not in self.file_results:
            self.file_results[rel] = {'passed': 0, 'failed': [], 'total': 0}

        self.file_results[rel]['total'] += 1

        if report.passed:
            self.file_results[rel]['passed'] += 1
            self.total_passed += 1
        elif report.failed:
            # Extract short error message
            short_msg = ''
            if report.longrepr:
                longrepr_str = str(report.longrepr)
                # Get the last line that looks like an assertion error
                lines = longrepr_str.strip().split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('_') and not line.startswith('='):
                        short_msg = line[:200]
                        break

            test_name = report.nodeid.split('::')[-1]
            self.file_results[rel]['failed'].append((test_name, short_msg))
            self.total_failed += 1
            self.failure_details.append((
                rel, test_name, short_msg, str(report.longrepr),
            ))


def get_default_seed():
    """Default seed: hash of container hostname."""
    hostname = os.uname().nodename
    return int(hashlib.md5(hostname.encode()).hexdigest()[:8], 16)


def collect_test_files(test_dir='tests'):
    """Collect all test_*.py files."""
    files = sorted(Path(test_dir).rglob('test_*.py'))
    return [str(f) for f in files]


def sample_tests(test_files, seed, fraction=0.1):
    """Deterministic sampling of test files."""
    rng = random.Random(seed)
    count = max(1, int(len(test_files) * fraction))
    return sorted(rng.sample(test_files, min(count, len(test_files))))


def main():
    parser = argparse.ArgumentParser(description='AsciiSwarm test runner')
    parser.add_argument('--fast', action='store_true',
                        help='Run deterministic 10%% sample of tests')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for --fast sampling (default: hash of hostname)')
    parser.add_argument('--log', type=str, default='test_failures.log',
                        help='Log file for verbose tracebacks (default: test_failures.log)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else get_default_seed()

    # Collect test files
    test_files = collect_test_files()
    if not test_files:
        print("No test files found.")
        return 1

    if args.fast:
        test_files = sample_tests(test_files, seed)
        print(f"[--fast] Running {len(test_files)} files (seed={seed})")

    # Run pytest with our collector plugin
    collector = QuietCollector()
    pytest_args = test_files + [
        '--override-ini=addopts=',  # clear any ini-level addopts
        '-q',
        '--no-header',
        '--tb=line',
    ]

    # Suppress all pytest output — redirect to devnull
    import io
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        ret = pytest.main(pytest_args, plugins=[collector])
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    # ---- Produce clean output ----
    for filepath in sorted(collector.file_results.keys()):
        result = collector.file_results[filepath]
        if result['failed']:
            print(f"FAIL {filepath}")
            for test_name, short_msg in result['failed']:
                print(f"  ERROR: {test_name} — {short_msg}")
        else:
            print(f"PASS {filepath} ({result['total']} tests)")

    total = collector.total_passed + collector.total_failed
    print(f"TOTAL: {collector.total_passed}/{total} passed")

    # ---- Write verbose log ----
    if collector.failure_details:
        with open(args.log, 'w') as f:
            for filepath, test_name, short_msg, traceback in collector.failure_details:
                f.write(f"{'='*60}\n")
                f.write(f"FAIL: {filepath}::{test_name}\n")
                f.write(f"{'='*60}\n")
                f.write(traceback)
                f.write('\n\n')
        print(f"\nVerbose tracebacks written to {args.log}")

    return 0 if collector.total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
