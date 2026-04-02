from __future__ import annotations

from dataclasses import dataclass

from app.session.session import SessionSummary


@dataclass(frozen=True)
class HumanSummary:
    title: str
    text: str


def build_popup_summary(summary: SessionSummary) -> HumanSummary:
    duration_s = summary.duration_s
    mins = int(duration_s // 60)
    secs = int(duration_s % 60)

    total_emotion_frames = sum(summary.emotion_counts.values())
    if total_emotion_frames > 0:
        top_label = max(summary.emotion_counts.items(), key=lambda kv: kv[1])[0]
    else:
        top_label = "—"

    lines: list[str] = []
    lines.append(f"Длительность: {mins:02d}:{secs:02d}")
    lines.append(f"Кадров обработано: {summary.frames_total}")
    lines.append(f"Кадров с лицами: {summary.frames_with_faces}")
    lines.append(f"Кадров без лиц: {summary.frames_no_faces}")
    lines.append("")
    lines.append("Распределение эмоций (по кадрам, где была классификация):")

    if total_emotion_frames == 0:
        lines.append("— не удалось классифицировать ни одного кадра (лица не найдены или качество низкое).")
    else:
        for label, cnt in sorted(summary.emotion_counts.items(), key=lambda kv: kv[1], reverse=True):
            if cnt <= 0:
                continue
            pct = 100.0 * cnt / float(total_emotion_frames)
            lines.append(f"- {label}: {pct:.1f}% ({cnt})")

        lines.append("")
        lines.append(f"Преобладающая эмоция: {top_label}")

    return HumanSummary(title="Итоги разговора", text="\n".join(lines))

