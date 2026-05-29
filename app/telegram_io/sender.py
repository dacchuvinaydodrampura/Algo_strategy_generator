"""
app/telegram_io/sender.py
--------------------------
Sends the generated PDF report and a short summary message back
to the Telegram channel.

Sends:
1. A short plaintext summary (verdict, key metrics, pattern count).
2. The full PDF as a document attachment.
3. Optionally a JSON summary file.

Error handling: if Telegram is unavailable, the report is still saved
locally and the error is logged — the pipeline never crashes on send failure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from telegram import Bot, InputFile
from telegram.error import TelegramError

from app.config import Settings
from app.models.session import BacktestResult
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_MAX_MESSAGE_LEN = 4096  # Telegram message character limit


def _build_summary_message(
    session_date_str: str,
    results: list[BacktestResult],
    archive_filename: str,
) -> str:
    """
    Build a concise Telegram summary message.
    Must stay under Telegram's 4096 character limit.
    """
    accepted = [r for r in results if r.verdict == "ACCEPTED"]
    marginal = [r for r in results if r.verdict == "MARGINAL"]
    rejected = [r for r in results if r.verdict == "REJECTED"]

    lines = [
        f"📊 *Market Research Report — {session_date_str}*",
        f"Archive: `{archive_filename}`",
        "",
        f"✅ Accepted patterns: *{len(accepted)}*",
        f"⚠️  Marginal patterns: *{len(marginal)}*",
        f"❌ Rejected patterns: *{len(rejected)}*",
        "",
    ]

    # Detail for each accepted/marginal pattern
    for result in accepted + marginal:
        icon = "✅" if result.verdict == "ACCEPTED" else "⚠️"
        wr = f"{result.win_rate:.1%}" if result.win_rate else "—"
        pf = f"{result.profit_factor:.2f}" if result.profit_factor else "—"
        n = result.sample_count
        lines.append(
            f"{icon} `{result.symbol}` {result.direction.value} | "
            f"WR={wr} PF={pf} n={n}"
        )
        if result.verdict == "MARGINAL":
            lines.append(f"   ↳ _{result.rejection_reason}_")

    if not accepted and not marginal:
        lines.append("_No viable patterns found for this session._")

    lines += [
        "",
        "_Full PDF report attached. This is not a trading recommendation._",
    ]

    msg = "\n".join(lines)
    # Trim if somehow over limit
    if len(msg) > _MAX_MESSAGE_LEN:
        msg = msg[: _MAX_MESSAGE_LEN - 20] + "\n…_(truncated)_"
    return msg


class TelegramSender:
    """
    Sends research reports back to the Telegram channel.

    Parameters
    ----------
    settings: Application settings (bot token, channel ID).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def send_report(
        self,
        pdf_path: Path,
        results: list[BacktestResult],
        session_date_str: str,
        archive_filename: str,
        json_summary_path: Optional[Path] = None,
    ) -> bool:
        """
        Send the summary message and PDF to Telegram.

        Returns True on success, False on failure.
        Never raises — all errors are logged.
        """
        bot = Bot(token=self._settings.telegram_bot_token)
        channel_id = self._settings.telegram_channel_id

        summary = _build_summary_message(session_date_str, results, archive_filename)

        # ── Send summary text ─────────────────────────────────────────────────
        try:
            await bot.send_message(
                chat_id=channel_id,
                text=summary,
                parse_mode="Markdown",
            )
            logger.info("telegram_summary_sent", date=session_date_str)
        except TelegramError as exc:
            logger.error("telegram_summary_send_failed", error=str(exc))
            return False

        # ── Send PDF ──────────────────────────────────────────────────────────
        if not pdf_path.exists():
            logger.error("pdf_not_found_for_send", path=str(pdf_path))
            return False

        try:
            with pdf_path.open("rb") as fh:
                await bot.send_document(
                    chat_id=channel_id,
                    document=InputFile(fh, filename=pdf_path.name),
                    caption=f"Research report for {session_date_str}",
                    read_timeout=120,
                    write_timeout=120,
                )
            logger.info("telegram_pdf_sent", path=str(pdf_path))
        except TelegramError as exc:
            logger.error(
                "telegram_pdf_send_failed",
                path=str(pdf_path),
                error=str(exc),
            )
            return False

        # ── Optionally send JSON summary ──────────────────────────────────────
        if json_summary_path and json_summary_path.exists():
            try:
                with json_summary_path.open("rb") as fh:
                    await bot.send_document(
                        chat_id=channel_id,
                        document=InputFile(fh, filename=json_summary_path.name),
                        caption="JSON metrics summary",
                    )
                logger.info("telegram_json_sent", path=str(json_summary_path))
            except TelegramError as exc:
                logger.warning(
                    "telegram_json_send_failed",
                    error=str(exc),
                )
                # Non-fatal: JSON is optional

        return True

    def send_report_sync(
        self,
        pdf_path: Path,
        results: list[BacktestResult],
        session_date_str: str,
        archive_filename: str,
        json_summary_path: Optional[Path] = None,
    ) -> bool:
        """Synchronous wrapper."""
        return asyncio.run(
            self.send_report(
                pdf_path=pdf_path,
                results=results,
                session_date_str=session_date_str,
                archive_filename=archive_filename,
                json_summary_path=json_summary_path,
            )
        )

    async def send_error_alert(self, session_date_str: str, error: str) -> None:
        """Send a brief error notification if the pipeline fails."""
        bot = Bot(token=self._settings.telegram_bot_token)
        msg = (
            f"🚨 *Research Engine Error — {session_date_str}*\n\n"
            f"```\n{error[:500]}\n```\n"
            "_Manual intervention may be required._"
        )
        try:
            await bot.send_message(
                chat_id=self._settings.telegram_channel_id,
                text=msg,
                parse_mode="Markdown",
            )
        except TelegramError as exc:
            logger.error("error_alert_send_failed", error=str(exc))

    async def send_status_alert(self, text: str) -> bool:
        """Send a general status alert or notification to the Telegram channel."""
        bot = Bot(token=self._settings.telegram_bot_token)
        try:
            await bot.send_message(
                chat_id=self._settings.telegram_channel_id,
                text=text,
                parse_mode="Markdown",
            )
            return True
        except TelegramError as exc:
            logger.error("telegram_status_alert_failed", error=str(exc))
            return False

    def send_status_alert_sync(self, text: str) -> bool:
        """Synchronous wrapper for send_status_alert."""
        return asyncio.run(self.send_status_alert(text))
