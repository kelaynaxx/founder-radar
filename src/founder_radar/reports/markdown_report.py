"""Markdown report generator.

Phase 1 produced a single Markdown document listing every collected
post, grouped by source and category, with engagement stats.

Phase 3 added a Top Opportunities section when opportunities are passed
in. When no opportunities exist, only the post section is rendered.

Design choices:
  - Pure string templating, no external Markdown library. The output is
    simple enough that adding a dependency would be overkill, and rolling
    our own keeps the report easy to tweak by hand.
  - Posts are grouped by `source` then `source_category`. With only
    Reddit in Phase 1 this means "all posts in r/entrepreneur" sits
    next to "all posts in r/startups", which is what humans want to skim.
  - Engagement is shown as `score / comments` so the reader can eyeball
    which threads attracted real discussion.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from founder_radar.reports.base import BaseReport

if TYPE_CHECKING:
    from founder_radar.database.models import Opportunity, Post


class MarkdownReport(BaseReport):
    """Render a Markdown summary of collected posts (and opportunities)."""

    extension: str = ".md"

    def render(
        self,
        posts: list["Post"],
        opportunities: list["Opportunity"] | None = None,
    ) -> str:
        """Build the full Markdown document as a single string."""
        lines: list[str] = []
        lines.append("# Founder Radar — Report")
        lines.append("")
        lines.append(
            f"_Generated: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        )
        lines.append(f"_Total posts: **{len(posts)}**_")
        if opportunities is not None:
            lines.append(
                f"_Total opportunities: **{len(opportunities)}**_"
            )
        lines.append("")

        if not posts:
            lines.append(
                "> No posts collected yet. Run `founder-radar collect` first."
            )
            return "\n".join(lines) + "\n"

        # Group: source -> category -> [posts]
        grouped: dict[str, dict[str | None, list["Post"]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for post in posts:
            grouped[post.source][post.source_category].append(post)

        # Render each source section.
        for source, by_category in sorted(grouped.items()):
            lines.append(f"## Source: `{source}`")
            lines.append("")
            for category, category_posts in sorted(
                by_category.items(), key=lambda kv: (kv[0] or "")
            ):
                heading = (
                    f"### {category}" if category else "### (uncategorized)"
                )
                lines.append(heading)
                lines.append("")
                lines.append(f"_{len(category_posts)} post(s)._")
                lines.append("")

                # Newest first within a category.
                for post in sorted(
                    category_posts,
                    key=lambda p: p.collected_at,
                    reverse=True,
                ):
                    lines.extend(self._render_post(post))
                    lines.append("")

            lines.append("")

        # Footer with running totals.
        lines.extend(self._render_totals(posts))

        # Phase 3: opportunities section.
        if opportunities:
            lines.append("")
            lines.extend(self._render_opportunities(opportunities))

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Per-post rendering
    # -------------------------------------------------------------------------
    def _render_post(self, post: "Post") -> list[str]:
        """Render a single post as a few Markdown lines."""
        lines: list[str] = []

        if post.url:
            lines.append(f"- **[{post.title}]({post.url})**")
        else:
            lines.append(f"- **{post.title}**")

        lines.append(
            f"  - score: `{post.score}` · comments: `{post.num_comments}` "
            f"· author: `{post.author or 'unknown'}`"
        )

        if post.body:
            excerpt = post.body.strip().replace("\n", " ")
            if len(excerpt) > 280:
                excerpt = excerpt[:277] + "..."
            lines.append(f"  - _{excerpt}_")

        return lines

    # -------------------------------------------------------------------------
    # Totals
    # -------------------------------------------------------------------------
    def _render_totals(self, posts: list["Post"]) -> list[str]:
        by_source: dict[str, int] = defaultdict(int)
        total_score = 0
        total_comments = 0
        for post in posts:
            by_source[post.source] += 1
            total_score += post.score
            total_comments += post.num_comments

        lines = ["---", "", "## Totals", ""]
        lines.append("| Source | Posts |")
        lines.append("|---|---:|")
        for source, count in sorted(by_source.items()):
            lines.append(f"| `{source}` | {count} |")
        lines.append(f"| **all** | **{len(posts)}** |")
        lines.append("")
        lines.append(f"- Total score across all posts: **{total_score}**")
        lines.append(f"- Total comments across all posts: **{total_comments}**")
        return lines

    # -------------------------------------------------------------------------
    # Opportunities (Phase 3)
    # -------------------------------------------------------------------------
    def _render_opportunities(
        self, opportunities: list["Opportunity"]
    ) -> list[str]:
        """Render the Top Opportunities section."""
        from founder_radar.database.connection import get_session
        from founder_radar.database.repository import OpportunityRepository

        lines: list[str] = []
        lines.append("---")
        lines.append("")
        lines.append("## Top opportunities (ranked)")
        lines.append("")

        if not opportunities:
            lines.append(
                "_No opportunities extracted yet. Run `founder-radar extract` "
                "after `founder-radar cluster` to populate this section._"
            )
            return lines

        for rank, opp in enumerate(opportunities, start=1):
            lines.append(f"### {rank}. {opp.title}")
            lines.append("")
            lines.append(
                f"_weighted: **{opp.weighted_score:.2f}** · "
                f"pain: **{opp.pain_score:.2f}** · "
                f"monetization: **{opp.monetization_score:.2f}** · "
                f"confidence: **{opp.confidence_score:.2f}** · "
                f"mentions: **{opp.mentions}** · "
                f"method: `{opp.extraction_method}` · "
                f"status: `{opp.status}`_"
            )
            lines.append("")
            lines.append(opp.problem_summary)
            lines.append("")
            if opp.target_audience:
                lines.append(f"**Audience:** {opp.target_audience}")
                lines.append("")

            lines.append("| factor | score |")
            lines.append("|---|---:|")
            lines.append(f"| frequency | {opp.frequency_score:.2f} |")
            lines.append(
                f"| emotional_intensity | {opp.emotional_intensity_score:.2f} |"
            )
            lines.append(
                f"| dissatisfaction | {opp.dissatisfaction_score:.2f} |"
            )
            lines.append(f"| market_size | {opp.market_size_score:.2f} |")
            lines.append(
                f"| ease_of_implementation | {opp.ease_of_implementation_score:.2f} |"
            )
            lines.append(
                f"| recurring_revenue | {opp.recurring_revenue_score:.2f} |"
            )
            lines.append(
                f"| technical_feasibility | {opp.technical_feasibility_score:.2f} |"
            )
            lines.append(f"| novelty | {opp.novelty_score:.2f} |")
            lines.append("")

            # Phase 3+ Reality Check + Trend line.
            sat_label = " SATURATED" if opp.saturation_score >= 0.7 else ""
            lines.append(
                f"_Reality check: saturation={opp.saturation_score:.2f}{sat_label}"
                f" · competitors={opp.distinct_competitor_count}"
                f" · trend=`{opp.trend}`_"
            )
            # Phase 3.5 Reality Validation: how viable is this opportunity?
            # We surface the status with an emoji-style indicator so the
            # reader can spot winners/losers at a glance. Reality is
            # ORTHOGONAL to ranking — high weighted_score + saturated = skip.
            reality_icons = {
                "underserved": "OPPORTUNITY",
                "competitive": "FRAGMENTED",
                "saturated":   "SATURATED",
                "unknown":     "UNKNOWN",
            }
            reality_icon = reality_icons.get(opp.reality_status, opp.reality_status)
            lines.append(
                f"_Reality view: **{reality_icon}**"
                f" (`{opp.reality_status}`,"
                f" confidence={opp.reality_confidence:.2f},"
                f" competitor_strength={opp.competitor_strength_estimate:.2f})_"
            )
            lines.append(
                f"_Reality check: saturation={opp.saturation_score:.2f}{sat_label}"
                f" · competitors={opp.distinct_competitor_count}"
                f" · trend=`{opp.trend}`_"
            )
            lines.append("")

            # JSON-list fields need decoding.
            with get_session() as session:
                tmp_repo = OpportunityRepository(session)
                ideas = tmp_repo.saas_ideas(opp)
                comps = tmp_repo.competitors(opp)
                links = tmp_repo.source_links(opp)

            if ideas:
                lines.append("**SaaS ideas:**")
                for idea in ideas:
                    lines.append(f"- {idea}")
                lines.append("")
            if comps:
                lines.append("**Existing competitors:**")
                for c in comps:
                    lines.append(f"- {c}")
                lines.append("")
            if links:
                lines.append(f"**Source posts ({len(links)}):**")
                for url in links[:5]:
                    lines.append(f"- {url}")
                if len(links) > 5:
                    lines.append(f"- ... and {len(links) - 5} more")
                lines.append("")

        return lines

