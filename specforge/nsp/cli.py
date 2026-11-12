"""Public import-light nsp command line."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

from specforge.nsp.store import (
    TERMINAL,
    RunStore,
    checkpoint_entries,
    read_json,
    resolve_checkpoint,
)


def public_value(value, key=""):
    lowered = key.lower().replace("-", "_")
    if lowered in {
        "wandb_key",
        "swanlab_key",
        "password",
        "secret",
        "api_key",
        "token",
    } or lowered.endswith(("_api_key", "_auth_token", "_password", "_secret")):
        return "<redacted>" if value is not None else None
    if isinstance(value, dict):
        return {name: public_value(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    return value


def emit(value):
    click.echo(json.dumps(public_value(value), indent=2, allow_nan=False))


def line(run):
    metric = run.get("latest_metrics", {})
    return f'{run["id"]}  {run["kind"]:5}  {run["state"]:10}  attempt={run["attempt"]}  step={metric.get("step", "—")}  {run.get("name") or ""}  {run.get("observation", "")}'


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--root",
    type=click.Path(path_type=Path),
    envvar="NSP_ROOT",
    help="Persistent task store (default ~/.local/share/nsp).",
)
@click.option(
    "--profile", help="Execute on a named SSH profile from ~/.config/nsp/profiles.yaml."
)
@click.pass_context
def cli(ctx, root, profile):
    """Launch and observe NexSpecTrainer training and evaluation runs."""
    ctx.obj = RunStore(root)


def submission_options(function):
    for option in reversed(
        [
            click.option("--name"),
            click.option(
                "--request-id",
                help="Retry the same submission safely using this stable ID.",
            ),
            click.option("--plan", is_flag=True),
            click.option("--watch", "watch_run", is_flag=True),
            click.option(
                "--foreground",
                is_flag=True,
                help="Wait; Ctrl-C requests stop (for scheduler-owned execution).",
            ),
            click.option("--json", "as_json", is_flag=True),
        ]
    ):
        function = option(function)
    return function


@cli.command()
@click.option("-c", "--config", type=click.Path(exists=True))
@click.option("--resume", help="Resume a terminal run from RUN@latest or RUN@step-N.")
@submission_options
@click.argument("overrides", nargs=-1)
@click.pass_obj
def train(
    store,
    config,
    resume,
    name,
    request_id,
    plan,
    watch_run,
    foreground,
    as_json,
    overrides,
):
    """Submit a typed training recipe; dotted section.field=value overrides are supported."""
    from specforge.nsp.manager import resume_config, submit, train_config

    if bool(config) == bool(resume):
        raise click.UsageError("provide exactly one of --config or --resume")
    resume_run = None
    if resume:
        if overrides or name or request_id:
            raise click.UsageError(
                "resume preserves experiment identity; overrides/name/request-id require a new run"
            )
        resume_run, resolved = resume_config(store, resume)
        launch = None
    else:
        resolved, launch = train_config(config, overrides)
    if plan:
        emit(
            {
                "kind": "train",
                "config": resolved,
                "launch": json.loads(launch.render()) if launch else None,
                "store": str(store.root),
                "execution": "local supervisor; run inside allocated GPU resources",
            }
        )
        return
    result = submit(
        store,
        "train",
        resolved,
        name=name,
        request_id=request_id,
        resume_run=resume_run,
    )
    emit(result) if as_json else click.echo(line(result))
    if watch_run or foreground:
        return monitor(
            store, result["id"], until=True, foreground=foreground, jsonl=as_json
        )


@cli.command("eval")
@click.argument("checkpoint")
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
@click.option("--metrics", help="Comma-separated loss,accuracy,prefix-al,al,performance; or all.")
@click.option("--group-by", type=click.Choice(["ctx-len"]))
@click.option("--ctx-bins", help="Left-closed edges, e.g. 0,4k,8k,16k,inf (k=1024).")
@submission_options
@click.pass_obj
def evaluate(
    store,
    checkpoint,
    config,
    metrics,
    group_by,
    ctx_bins,
    name,
    request_id,
    plan,
    watch_run,
    foreground,
    as_json,
):
    """Submit an independent evaluation of a fixed checkpoint."""
    from specforge.nsp.eval_config import load_eval
    from specforge.nsp.manager import submit

    fixed = resolve_checkpoint(store, checkpoint)
    spec = load_eval(config, fixed, metrics, group_by, ctx_bins)
    if plan:
        emit(
            {
                "kind": "eval",
                "config": spec.model_dump(mode="json"),
                "stages": [
                    stage
                    for stage, yes in [
                        ("offline", any(m not in ("al", "performance") for m in spec.metrics)),
                        ("serving", any(m in ("al", "performance") for m in spec.metrics)),
                    ]
                    if yes
                ],
            }
        )
        return
    result = submit(
        store, "eval", spec.model_dump(mode="json"), name=name, request_id=request_id
    )
    emit(result) if as_json else click.echo(line(result))
    if watch_run or foreground:
        return monitor(
            store, result["id"], until=True, foreground=foreground, jsonl=as_json
        )


@cli.command("calibrate")
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
@submission_options
@click.pass_obj
def calibrate(store, config, name, request_id, plan, watch_run, foreground, as_json):
    """Fit a frozen n-gram calibrator from deterministic target trajectories."""
    from specforge.nsp.calibrate_config import load_calibration
    from specforge.nsp.manager import submit

    spec = load_calibration(config)
    if plan:
        emit({"kind": "calibrate", "config": spec.model_dump(mode="json")})
        return
    result = submit(store, "calibrate", spec.model_dump(mode="json"), name=name, request_id=request_id)
    emit(result) if as_json else click.echo(line(result))
    if watch_run or foreground:
        return monitor(store, result["id"], until=True, foreground=foreground, jsonl=as_json)


@cli.command("list")
@click.option("--kind", type=click.Choice(["train", "eval", "calibrate"]))
@click.option("--status")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def list_runs(store, kind, status, as_json):
    """List submitted runs, including their latest attempt and observation freshness."""
    runs = [
        run
        for run in store.list()
        if (kind is None or kind == run["kind"])
        and (status is None or status == run["state"])
    ]
    (
        emit({"schema_version": 1, "runs": runs})
        if as_json
        else click.echo("\n".join(line(r) for r in runs) or "No runs.")
    )


@cli.command()
@click.argument("run")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def show(store, run, as_json):
    """Show a run's latest attempt, source identity, configuration and metrics."""
    value = store.status(run)
    value["config"] = read_json(store.attempt(run) / "resolved.json")
    value["manifest"] = read_json(store.attempt(run) / "manifest.json")
    emit(value)


def monitor(
    store,
    run=None,
    *,
    until=False,
    foreground=False,
    jsonl=False,
    interval=2,
    timeout=None,
):
    start = time.monotonic()
    previous = None
    previous_term = None
    if foreground:

        def interrupted(*_):
            raise KeyboardInterrupt

        previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            value = (
                store.status(run)
                if run
                else {"runs": [r for r in store.list() if r["state"] not in TERMINAL]}
            )
            snapshot = json.dumps(value, sort_keys=True)
            if snapshot != previous:
                if jsonl:
                    click.echo(json.dumps(value, allow_nan=False))
                elif run:
                    click.echo(line(value))
                    metrics = value.get("latest_metrics")
                    if metrics:
                        click.echo(json.dumps(metrics, allow_nan=False))
                else:
                    click.echo(
                        "\n".join(line(r) for r in value["runs"]) or "No active runs."
                    )
                previous = snapshot
            if run and until and value["state"] in TERMINAL:
                return {"succeeded": 0, "failed": 1, "cancelled": 4}[value["state"]]
            if timeout is not None and time.monotonic() - start >= timeout:
                return 3
            time.sleep(interval)
    except KeyboardInterrupt:
        if foreground and run:
            (store.attempt(run) / "stop.request").touch()
            return monitor(store, run, until=True, jsonl=jsonl, interval=interval)
        return 0
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


@cli.command()
@click.argument("run", required=False)
@click.option("--until", type=click.Choice(["terminal"]))
@click.option("--interval", type=click.FloatRange(min=0.1), default=2)
@click.option("--timeout", type=click.FloatRange(min=0.1))
@click.option("--jsonl", is_flag=True)
@click.pass_obj
def watch(store, run, until, interval, timeout, jsonl):
    """Observe a run, or all active runs. Ctrl-C only detaches observation."""
    if until and not run:
        raise click.UsageError("--until requires a run ID")
    return monitor(
        store, run, until=bool(until), interval=interval, timeout=timeout, jsonl=jsonl
    )


@cli.command()
@click.argument("run")
@click.option("--follow", is_flag=True)
@click.pass_obj
def logs(store, run, follow):
    """Read the latest attempt's combined worker log."""
    attempt = store.attempt(run)
    offset = 0
    while True:
        path = attempt / "worker.log"
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                click.echo(stream.read(), nl=False)
                offset = stream.tell()
        if not follow or store.status(run)["state"] in TERMINAL:
            return
        time.sleep(0.5)


@cli.command()
@click.argument("run")
@click.pass_obj
def stop(store, run):
    """Request graceful termination of precisely this run's latest attempt."""
    with store.lock():
        value = store.status(run)
        if value["state"] not in TERMINAL:
            (store.attempt(run) / "stop.request").touch()
    emit(
        {
            "run_id": run,
            "state": value["state"],
            "stop_requested": value["state"] not in TERMINAL,
        }
    )


@cli.command()
@click.argument("run")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def checkpoints(store, run, as_json):
    """List completed checkpoint publications across all attempts."""
    entries = []
    for attempt in sorted((store.path(run) / "attempts").iterdir()):
        config = read_json(attempt / "resolved.json")
        if "output_dir" in config:
            entries.extend(
                {**entry, "attempt": attempt.name}
                for entry in checkpoint_entries(config["output_dir"])
            )
    emit({"schema_version": 1, "checkpoints": entries})


@cli.command()
@click.argument("run")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def results(store, run, as_json):
    """Show overall and ctx-bin results, including partial results of failed attempts."""
    path = store.attempt(run) / "results.json"
    if not path.exists():
        raise click.ClickException("no evaluation results published yet")
    value = read_json(path)
    if as_json:
        emit(value)
        return
    click.echo(f'complete={value["complete"]}')
    if value.get("error"):
        click.echo(value["error"])
    if value.get("calibrator"):
        click.echo(f'calibrator={value["calibrator"]}')
        emit(value["fit"])
    for stage, result in value["stages"].items():
        click.echo(stage)
        rows = [("overall", result["overall"])] + [
            (
                f'[{b["lower"]},{b["upper"] if b["upper"] is not None else "inf"})',
                b["metrics"],
            )
            for b in result["bins"]
        ]
        for label, metrics in rows:
            cells = [
                (
                    f'{name}={m["value"]:.6f} (n={m["denominator"]:g})'
                    if m["value"] is not None
                    else f"{name}=— (n=0)"
                )
                for name, m in metrics.items()
            ]
            click.echo(f"{label:20} " + "  ".join(cells))


def remote_arguments(argv):
    """SSH profiles forward the same CLI; configs/data paths refer to the remote host."""
    if "--profile" not in argv:
        return None
    index = argv.index("--profile")
    if index + 1 >= len(argv):
        raise ValueError("--profile requires a name")
    name = argv[index + 1]
    rest = argv[:index] + argv[index + 2 :]
    import yaml

    profiles = Path(
        os.environ.get("NSP_PROFILES", "~/.config/nsp/profiles.yaml")
    ).expanduser()
    profile = yaml.safe_load(profiles.read_text())[name]
    if set(profile) - {"host", "root", "cwd", "python"}:
        raise ValueError("SSH profile fields: host, root, cwd, python")
    if "--root" in rest:
        raise ValueError("--root belongs in the selected profile")
    command = [
        profile.get("python", "python3"),
        "-m",
        "specforge.nsp.cli",
        "--root",
        profile["root"],
        *rest,
    ]
    return [
        "ssh",
        "--",
        profile["host"],
        f'cd {shlex.quote(profile["cwd"])} && exec {shlex.join(command)}',
    ]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if sys.version_info < (3, 11):
        click.echo("Error: nsp requires Python 3.11 or later", err=True)
        return 2
    try:
        remote = remote_arguments(argv)
        if remote:
            return subprocess.call(remote)
        return int(cli.main(args=argv, prog_name="nsp", standalone_mode=False) or 0)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except (ValueError, OSError, KeyError) as exc:
        click.echo(f"Error: {exc}", err=True)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
