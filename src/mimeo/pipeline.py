"""Orchestrator: run all stages end-to-end for one expert."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel

from .avatar import generate_avatar
from .config import Settings, ensure_dirs
from .critique import critique_agents, critique_skill
from .discovery import discover_sources
from .distill import distill_all
from .fetchers import fetch_all
from .identity import resolve_identity
from .llm import LLMClient
from .parallel_client import ParallelClient
from .research import deep_research
from .schemas import Extraction, FetchedContent, Source
from .synthesize import author_agents, author_skill, cluster_corpus
from .verify import verify_quotes
from .writers import write_agents, write_skill

logger = logging.getLogger(__name__)


async def run_pipeline(
    settings: Settings,
    *,
    console: Console | None = None,
    on_stage: Callable[[str, str], None] | None = None,
    parallel: ParallelClient | None = None,
    llm: LLMClient | None = None,
) -> Path:
    """Run the whole pipeline and return the path to the generated skill.

    ``parallel`` and ``llm`` are injectable so tests can pass fakes. In real
    use both default to freshly constructed clients that read their API keys
    from the environment.
    """

    console = console or Console()
    ensure_dirs(settings)

    def stage(name: str, detail: str = "") -> None:
        console.rule(f"[bold cyan]{name}")
        if detail:
            console.print(detail)
        if on_stage:
            on_stage(name, detail)

    expert_line = settings.expert_name
    if settings.expert_description:
        expert_line = f"{settings.expert_name} ({settings.expert_description})"
    console.print(
        Panel(
            (
                f"[bold]Expert:[/bold] {expert_line}\n"
                f"[bold]Format:[/bold] {settings.format}\n"
                f"[bold]Mode:[/bold] {settings.mode}\n"
                f"[bold]Max sources:[/bold] {settings.max_sources}\n"
                f"[bold]Deep research:[/bold] {'yes' if settings.deep_research else 'no'}\n"
                f"[bold]Verify quotes:[/bold] {'yes' if settings.verify_quotes else 'no'}\n"
                f"[bold]Critique:[/bold] {'yes' if settings.critique else 'no'}\n"
                f"[bold]Avatar:[/bold] {'yes (' + settings.avatar_model + ')' if settings.generate_avatar else 'no'}\n"
                f"[bold]Model:[/bold] {settings.model}\n"
                f"[bold]Output:[/bold] {settings.skill_dir}"
            ),
            title="mimeo",
            border_style="cyan",
        )
    )

    if parallel is None:
        parallel = ParallelClient()
    if llm is None:
        llm = LLMClient(model=settings.model, provider=settings.provider)

    # Stage 0: disambiguate the name before spending money on discovery.
    # Cheap (one search + one LLM call) and short-circuits with an error
    # instead of silently blending two different people's work.
    settings = await resolve_identity(
        settings=settings, parallel=parallel, llm=llm, console=console
    )

    write_skill_flag = settings.format in ("skill", "both")
    write_agents_flag = settings.format in ("agents", "both")
    # Baseline = discover, fetch, distill, cluster. Each output artifact adds
    # one authoring step, plus an optional critique step per artifact.
    # Verify-quotes is a side-step and doesn't count. Deep research is the
    # same.
    total_stages = (
        4
        + int(write_skill_flag)
        + int(write_agents_flag)
        + (int(write_skill_flag) + int(write_agents_flag)) * int(settings.critique)
    )

    stage(
        f"1/{total_stages} Discovery",
        "Searching across essays, talks, interviews, podcasts, frameworks, books...",
    )
    sources: list[Source] = await discover_sources(
        settings=settings, parallel=parallel, llm=llm
    )
    console.print(f"Selected [bold]{len(sources)}[/bold] sources.")
    if not sources:
        raise RuntimeError("No sources discovered. Check PARALLEL_API_KEY and the expert name.")

    stage(f"2/{total_stages} Fetch", "Fetching full content for each source...")
    fetched: list[FetchedContent] = await fetch_all(
        sources, settings=settings, parallel=parallel
    )
    console.print(
        f"Fetched content for [bold]{len(fetched)}[/bold] / {len(sources)} sources "
        f"({sum(f.char_count for f in fetched):,} chars total)."
    )

    if settings.deep_research:
        stage(
            f"2.5/{total_stages} Deep research",
            "Running Parallel Task API pro-fast (this can take a few minutes)...",
        )
        pair = await deep_research(settings=settings, parallel=parallel)
        if pair:
            research_source, research_content = pair
            sources.append(research_source)
            fetched.append(research_content)
            console.print(
                f"Deep-research report added as [bold]{research_source.id}[/bold] "
                f"({research_content.char_count:,} chars)."
            )
        else:
            console.print("[yellow]Deep research failed or returned empty; continuing without it.[/yellow]")

    stage(
        f"3/{total_stages} Distill",
        "Extracting principles, frameworks, and quotes from each source...",
    )
    extractions: list[Extraction] = await distill_all(
        sources=sources, fetched=fetched, settings=settings, llm=llm
    )
    console.print(
        f"Distilled [bold]{len(extractions)}[/bold] sources into structured extractions."
    )

    stage(f"4/{total_stages} Cluster", "Merging extractions into a unified corpus...")
    corpus = await cluster_corpus(
        extractions=extractions, settings=settings, llm=llm
    )
    console.print(
        f"Clustered into {len(corpus.principles)} principles, "
        f"{len(corpus.frameworks)} frameworks, "
        f"{len(corpus.mental_models)} mental models, "
        f"{len(corpus.signature_quotes)} quotes."
    )

    if settings.verify_quotes:
        console.rule("[bold cyan]Verifying quotes")
        corpus, verify_report = verify_quotes(
            corpus=corpus, fetched=fetched, settings=settings
        )
        if verify_report.total == 0:
            console.print("[dim]No quotes to verify.[/dim]")
        else:
            pass_pct = verify_report.pass_rate * 100
            style = "green" if verify_report.pass_rate >= 0.9 else "yellow"
            console.print(
                f"[{style}]{verify_report.verified}/{verify_report.total} "
                f"quotes verified ({pass_pct:.0f}%).[/{style}]"
            )
            if verify_report.unverified:
                console.print(
                    f"[yellow]Stripped {len(verify_report.unverified)} "
                    "unverified quote(s); report in "
                    "_workspace/quote_verification.md.[/yellow]"
                )
        # Re-persist the cleaned corpus so downstream authoring and any
        # ``--refresh`` re-runs see the post-verification state.
        cluster_cache = (
            settings.workspace_dir / f"clustered_corpus.{settings.model_cache_id}.json"
        )
        cluster_cache.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")

    authoring_index = 5

    written: list[str] = []

    if write_skill_flag:
        attempt = 1
        max_attempts = 3
        feedback = None
        skill_output = None
        skill_path = None
        
        while attempt <= max_attempts:
            stage(
                f"{authoring_index}/{total_stages} Author skill" + (f" (Attempt {attempt}/{max_attempts})" if settings.critique else ""),
                "Writing SKILL.md and references/*.md..." if attempt == 1 else f"Re-authoring SKILL.md based on critique feedback (Attempt {attempt}/{max_attempts})...",
            )
            skill_output = await author_skill(
                corpus=corpus, settings=settings, llm=llm, feedback=feedback
            )
            skill_path = write_skill(
                output=skill_output, sources=sources, settings=settings
            )
            
            if not settings.critique:
                break
                
            stage(
                f"{authoring_index + 1}/{total_stages} Critique skill (Attempt {attempt}/{max_attempts})",
                "Adversarial review of the authored skill...",
            )
            report = await critique_skill(
                output=skill_output, corpus=corpus, settings=settings, llm=llm, write_report=True
            )
            console.print(_critique_summary(report, label="SKILL.md"))
            
            if report.overall_score >= 8:
                console.print(f"[bold green]SKILL.md passed critique with score {report.overall_score}/10![/bold green]")
                break
                
            if attempt == max_attempts:
                console.print(f"[bold yellow]SKILL.md critique score ({report.overall_score}/10) is below threshold, but maximum retries reached.[/bold yellow]")
                break
                
            issues_text = []
            for idx, issue in enumerate(report.issues, 1):
                suggestion_str = f" Suggestion: {issue.suggestion}" if issue.suggestion else ""
                issues_text.append(f"{idx}. [{issue.severity.upper()}] in {issue.location}: {issue.description}{suggestion_str}")
            
            feedback = (
                f"Overall Score: {report.overall_score}/10\n"
                f"Summary of issues:\n" + "\n".join(issues_text)
            )
            console.print(f"[bold yellow]SKILL.md score is low ({report.overall_score}/10). Initiating automated rewrite loop...[/bold yellow]")
            attempt += 1

        written.append(f"SKILL.md + references/ at [bold green]{skill_path}[/bold green]")
        authoring_index += 2 if settings.critique else 1

    if write_agents_flag:
        attempt = 1
        max_attempts = 3
        feedback = None
        agents_output = None
        agents_path = None
        
        while attempt <= max_attempts:
            stage(
                f"{authoring_index}/{total_stages} Author AGENTS.md" + (f" (Attempt {attempt}/{max_attempts})" if settings.critique else ""),
                "Writing AGENTS.md..." if attempt == 1 else f"Re-authoring AGENTS.md based on critique feedback (Attempt {attempt}/{max_attempts})...",
            )
            agents_output = await author_agents(
                corpus=corpus, settings=settings, llm=llm, feedback=feedback
            )
            agents_path = write_agents(
                output=agents_output, sources=sources, settings=settings
            )
            
            if not settings.critique:
                break
                
            stage(
                f"{authoring_index + 1}/{total_stages} Critique AGENTS.md (Attempt {attempt}/{max_attempts})",
                "Adversarial review of the authored AGENTS.md...",
            )
            report = await critique_agents(
                output=agents_output, corpus=corpus, settings=settings, llm=llm, write_report=True
            )
            console.print(_critique_summary(report, label="AGENTS.md"))
            
            if report.overall_score >= 8:
                console.print(f"[bold green]AGENTS.md passed critique with score {report.overall_score}/10![/bold green]")
                break
                
            if attempt == max_attempts:
                console.print(f"[bold yellow]AGENTS.md critique score ({report.overall_score}/10) is below threshold, but maximum retries reached.[/bold yellow]")
                break
                
            issues_text = []
            for idx, issue in enumerate(report.issues, 1):
                suggestion_str = f" Suggestion: {issue.suggestion}" if issue.suggestion else ""
                issues_text.append(f"{idx}. [{issue.severity.upper()}] in {issue.location}: {issue.description}{suggestion_str}")
            
            feedback = (
                f"Overall Score: {report.overall_score}/10\n"
                f"Summary of issues:\n" + "\n".join(issues_text)
            )
            console.print(f"[bold yellow]AGENTS.md score is low ({report.overall_score}/10). Initiating automated rewrite loop...[/bold yellow]")
            attempt += 1

        written.append(f"AGENTS.md at [bold green]{agents_path}[/bold green]")
        authoring_index += 2 if settings.critique else 1

    if settings.generate_avatar:
        console.rule("[bold cyan]Avatar")
        console.print(
            f"Generating avatar with [bold]{settings.avatar_model}[/bold]..."
        )
        try:
            avatar_path = await generate_avatar(settings=settings)
        except Exception as exc:  # noqa: BLE001 - avatar is best-effort
            logger.warning("Avatar generation failed: %s", exc)
            console.print(
                f"[yellow]Avatar generation failed ({exc}); continuing.[/yellow]"
            )
        else:
            if avatar_path is not None:
                console.print(f"Avatar saved to [bold green]{avatar_path}[/bold green]")
                written.append(f"avatar at [bold green]{avatar_path}[/bold green]")
            else:
                prompt_path = settings.skill_dir / "avatar_prompt.txt"
                if prompt_path.exists():
                    console.print(
                        f"[yellow]Automated avatar skipped/failed. "
                        f"Image generation prompt written to [bold]{prompt_path}[/bold] "
                        "for external use.[/yellow]"
                    )
                    written.append(f"avatar prompt at [bold green]{prompt_path}[/bold green]")
                else:
                    console.print(
                        "[yellow]Avatar model returned no image; skipping.[/yellow]"
                    )

    console.print(
        Panel(
            "\n".join(written) if written else "[yellow]Nothing written.[/yellow]",
            title="Done",
            border_style="green",
        )
    )
    return settings.skill_dir


def _critique_summary(report, *, label: str) -> str:  # type: ignore[no-untyped-def]
    """One-line console summary for a critique pass."""
    highs = sum(1 for i in report.issues if i.severity == "high")
    mediums = sum(1 for i in report.issues if i.severity == "medium")
    colour = "green" if report.overall_score >= 8 else "yellow" if report.overall_score >= 6 else "red"
    return (
        f"[{colour}]{label} score: {report.overall_score}/10[/{colour}] "
        f"— {highs} high, {mediums} medium issues. "
        "Full report in _workspace/."
    )
