# airflow/dags/life_os_weekly_synthesis.py
"""
Weekly Synthesis DAG
─────────────────────────────────────────────────────────────────────────────
Runs every Sunday at 11:00 PM to synthesize the week (Mon–Sun) just ending.
Generates an AI pattern summary from NUMERIC DATA ONLY.

PRIVACY ARCHITECTURE — HARD CONSTRAINT:
    content, gratitude, and challenges from journal_entries are NEVER
    collected or included in any API call. The AI sees only:
      - mood_score values (integers 1–5)
      - energy_score values (integers 1–5)
      - habit completion counts (when Phase 2 is built)
      - workout metrics (Phase 4)

AI Provider:
    Default: Gemini 2.5 Flash via _gemini_flash() in blog_agents.py
    Optional: Ollama (local) — set SYNTHESIS_AI_PROVIDER=ollama in the
              Airflow Variable store or environment. The Ollama path is
              stubbed and ready to activate without code changes.

Task DAG:
    task_collect_data
        >> task_generate_synthesis
        >> task_save_synthesis
        >> task_lock_old_entries

Schedule: 0 23 * * 0  (Sunday 11 PM)
"""

import sys
import logging
from datetime import datetime, timedelta, date

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── AI PROVIDER CONFIGURATION ──────────────────────────────────────────────────
_DEFAULT_AI_PROVIDER = "gemini"
_OLLAMA_MODEL = "llama3.1:8b"           
_OLLAMA_ENDPOINT = "http://ollama:11434/api/generate"
_GEMINI_MODEL_LABEL = "gemini-2.5-flash"

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

# ── HELPER FUNCTION ────────────────────────────────────────────────────────────

def _format_workout_section(workout_data: dict) -> str:
    """Format workout data for the synthesis prompt."""
    if not workout_data or workout_data.get("count", 0) == 0:
        return "No workout sessions logged this week."

    lines = [f"- Sessions completed: {workout_data['count']}"]

    if workout_data.get("total_minutes"):
        h, m = divmod(workout_data["total_minutes"], 60)
        duration_str = f"{int(h)}h {int(m)}m" if h else f"{int(m)}m"
        lines.append(f"- Total training time: {duration_str}")

    if workout_data.get("avg_fatigue") is not None:
        fatigue_label = {
            1: "Easy", 2: "Light", 3: "Moderate", 4: "Hard", 5: "Brutal"
        }.get(round(workout_data["avg_fatigue"]), "Moderate")
        lines.append(f"- Average fatigue: {workout_data['avg_fatigue']}/5 ({fatigue_label})")

    bw_start = workout_data.get("body_weight_start")
    bw_end = workout_data.get("body_weight_end")
    unit = workout_data.get("body_weight_unit", "lb")
    
    if bw_start and bw_end:
        delta = round(bw_end - bw_start, 1)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        lines.append(f"- Body weight: {bw_start} → {bw_end} {unit} ({delta_str} {unit})")
    elif bw_end:
        lines.append(f"- Body weight: {bw_end} {unit}")

    if workout_data.get("muscle_groups_trained"):
        top_muscles = [r["muscle"].replace("_", " ") for r in workout_data["muscle_groups_trained"][:4]]
        lines.append(f"- Muscle groups trained: {', '.join(top_muscles)}")

    return "\n".join(lines)


# ── TASK 1: Collect numeric data ───────────────────────────────────────────────

def task_collect_data(**context):
    """
    Gathers only numeric metrics for the week just ended.
    NEVER touches content, gratitude, or challenges columns.
    Pushes structured data dict to XCom for the synthesis task.
    """
    from dag_db import fetch_all, fetch_one

    # Calculate week boundaries (Mon–Sun ending last Sunday)
    today = date.today()
    days_since_sunday = (today.weekday() + 1) % 7 
    week_end = today - timedelta(days=days_since_sunday) if days_since_sunday > 0 else today
    week_start = week_end - timedelta(days=6)

    log.info("Collecting data for week %s to %s", week_start, week_end)

    # ── MOOD AND ENERGY SCORES ONLY ─────────────────────────────────────────────
    journal_rows = fetch_all(
        """SELECT entry_date, mood_score, energy_score
           FROM journal_entries
           WHERE entry_date BETWEEN %s AND %s
           ORDER BY entry_date""",
        (week_start.isoformat(), week_end.isoformat())
    )

    daily_scores = {}
    mood_scores = []
    energy_scores = []
    for row in journal_rows:
        d = row['entry_date']
        mood = row['mood_score']
        energy = row['energy_score']
        daily_scores[str(d)] = {"mood": mood, "energy": energy}
        if mood is not None:
            mood_scores.append(mood)
        if energy is not None:
            energy_scores.append(energy)

    avg_mood = round(sum(mood_scores) / len(mood_scores), 2) if mood_scores else None
    avg_energy = round(sum(energy_scores) / len(energy_scores), 2) if energy_scores else None
    data_sources = ["journal_entries"]

    # ── HABIT DATA (Phase 2) ──────────────────────────────────────────────────
    habits_completion_rate = None
    habit_summary = {}

    try:
        habit_rows = fetch_all(
            """SELECT h.name, COUNT(hl.id) AS completions, 7 AS possible
               FROM habits h
               LEFT JOIN habit_logs hl
                   ON hl.habit_id = h.id
                   AND hl.log_date BETWEEN %s AND %s
                   AND hl.completed = 1
               WHERE h.is_active = 1
               GROUP BY h.id, h.name""",
            (week_start.isoformat(), week_end.isoformat())
        )
        if habit_rows:
            total_possible = sum(r['possible'] for r in habit_rows)
            total_completed = sum(r['completions'] for r in habit_rows)
            habits_completion_rate = round(
                (total_completed / total_possible * 100) if total_possible else 0, 2
            )
            habit_summary = {
                r['name']: {"completed": r['completions'], "possible": r['possible']}
                for r in habit_rows
            }
            data_sources.append("habit_logs")
            log.info("Habit data collected: %d habits, %.1f%% completion", len(habit_rows), habits_completion_rate)
    except Exception as exc:
        log.info("Habit data skipped (Phase 2 not yet built): %s", exc)
        

    # ── WEEKLY PLAN ADHERENCE ────────────────────────────────────────────────────
    plan_adherence = {}
    try:
        plan_row = fetch_one(
            """SELECT meal_adherence_pct, workout_adherence_pct,
                      meals_planned, meals_followed,
                      workouts_planned, workouts_completed
               FROM weekly_plans
               WHERE week_start_date = %s""",
            (week_start.isoformat(),)
        )
        if plan_row:
            plan_adherence = dict(plan_row)
            data_sources.append("weekly_plans")
            log.info(
                "Plan adherence: meals %d%%, workouts %d%%",
                plan_row.get("meal_adherence_pct", 0),
                plan_row.get("workout_adherence_pct", 0),
            )
    except Exception as exc:
        log.info("Weekly plan adherence skipped: %s", exc)

    # Add to structured_data:
    # structured_data["plan_adherence"] = plan_adherence

    # ── WORKOUT DATA (Phase 4) ────────────────────────────────────────────────
    try:
        workouts = fetch_all(
            "SELECT session_date, duration_minutes, fatigue_rating "
            "FROM workout_sessions "
            "WHERE session_date BETWEEN %s AND %s "
            "AND ended_at IS NOT NULL",
            (week_start.isoformat(), week_end.isoformat())
        )

        body_start = fetch_one(
            "SELECT weight, weight_unit FROM body_metrics "
            "WHERE metric_date >= %s ORDER BY metric_date ASC LIMIT 1",
            (week_start.isoformat(),)
        )
        body_end = fetch_one(
            "SELECT weight, weight_unit FROM body_metrics "
            "WHERE metric_date <= %s ORDER BY metric_date DESC LIMIT 1",
            (week_end.isoformat(),)
        )

        muscle_freq = fetch_all(
            "SELECT e.primary_muscle_group, COUNT(ws.id) AS set_count "
            "FROM workout_sets ws "
            "JOIN exercises e ON ws.exercise_id = e.id "
            "JOIN workout_sessions sess ON ws.session_id = sess.id "
            "WHERE sess.session_date BETWEEN %s AND %s "
            "  AND ws.is_warmup = 0 "
            "  AND sess.ended_at IS NOT NULL "
            "GROUP BY e.primary_muscle_group "
            "ORDER BY set_count DESC",
            (week_start.isoformat(), week_end.isoformat())
        )

        workout_data = {
            "count": len(workouts),
            "total_minutes": sum(w["duration_minutes"] or 0 for w in workouts),
            "avg_fatigue": (
                round(sum(w["fatigue_rating"] or 0 for w in workouts) / len(workouts), 1)
                if workouts else None
            ),
            "body_weight_start": float(body_start["weight"]) if body_start and body_start.get("weight") else None,
            "body_weight_end": float(body_end["weight"]) if body_end and body_end.get("weight") else None,
            "body_weight_unit": body_end["weight_unit"] if body_end else "lb",
            "muscle_groups_trained": [
                {"muscle": r["primary_muscle_group"], "sets": r["set_count"]}
                for r in muscle_freq
            ],
        }

        if workout_data["count"] > 0:
            data_sources.append("workout_sessions")
            log.info("Workout data collected: %d sessions", workout_data["count"])

    except Exception as e:
        log.warning(f"Failed to collect workout data: {e}")
        workout_data = {"count": 0}

    structured_data = {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "days_logged": len(journal_rows),
        "daily_scores": daily_scores,
        "avg_mood": avg_mood,
        "avg_energy": avg_energy,
        "mood_scores": mood_scores,
        "energy_scores": energy_scores,
        "habits_completion_rate": habits_completion_rate,
        "habit_summary": habit_summary,
        "workout_count": workout_data.get("count", 0),
        "workout_data": workout_data,
        "data_sources": data_sources,
        "plan_adherence": plan_adherence,
    }

    log.info(
        "Data collected: %d days, avg_mood=%s, avg_energy=%s",
        len(journal_rows), avg_mood, avg_energy
    )

    context['ti'].xcom_push(key='structured_data', value=structured_data)


# ── TASK 2: Generate synthesis ─────────────────────────────────────────────────

def task_generate_synthesis(**context):
    """
    Builds the synthesis prompt from numeric data ONLY and calls the AI.
    Never includes any journal text content.
    """
    import os

    structured_data = context['ti'].xcom_pull(key='structured_data', task_ids='collect_data')
    if not structured_data:
        raise ValueError("No structured data received from collect_data task")

    if structured_data['days_logged'] == 0:
        log.info("No journal entries for the week — skipping synthesis generation")
        context['ti'].xcom_push(key='synthesis_text', value=None)
        context['ti'].xcom_push(key='model_used', value=None)
        return

    # ── BUILD PROMPT FROM NUMBERS ONLY ────────────────────────────────────────
    week_start = structured_data['week_start']
    week_end = structured_data['week_end']
    daily_scores = structured_data['daily_scores']
    habits = structured_data.get('habit_summary', {})
    habits_rate = structured_data.get('habits_completion_rate')
    workout_data = structured_data.get('workout_data', {})

    # Format daily score table
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    score_lines = []
    sorted_dates = sorted(daily_scores.keys())
    for i, date_str in enumerate(sorted_dates):
        scores = daily_scores[date_str]
        day_label = day_names[i] if i < 7 else date_str
        mood_str = str(scores['mood']) if scores['mood'] is not None else 'not logged'
        energy_str = str(scores['energy']) if scores['energy'] is not None else 'not logged'
        score_lines.append(f"  {day_label}: mood={mood_str}, energy={energy_str}")

    scores_block = "\n".join(score_lines) if score_lines else "  No entries this week"

    # Format habit summary
    habit_lines = []
    for habit_name, data in habits.items():
        pct = round(data['completed'] / data['possible'] * 100) if data['possible'] else 0
        habit_lines.append(f"  - {habit_name}: {data['completed']}/{data['possible']} days ({pct}%)")
    habits_block = (
        "\n".join(habit_lines) if habit_lines
        else "  No habit data available (habits module not yet set up)"
    )

    workout_section = _format_workout_section(workout_data)

    prompt = f"""Week of {week_start} to {week_end}:

MOOD AND ENERGY SCORES (1-5 scale):
{scores_block}

Weekly averages:
  - Average mood: {structured_data['avg_mood'] or 'no data'}/5
  - Average energy: {structured_data['avg_energy'] or 'no data'}/5
  - Days with entries: {structured_data['days_logged']}/7

HABITS:
{habits_block}
{"  Overall completion rate: " + str(habits_rate) + "%" if habits_rate is not None else ""}

## Workouts This Week
{workout_section}

Generate a synthesis for this week."""

    adherence = structured_data.get("plan_adherence", {})
    if adherence:
        meal_pct = adherence.get("meal_adherence_pct") or adherence.get(
            "meals_followed", 0
        ) / max(adherence.get("meals_planned", 1), 1) * 100
        workout_pct = adherence.get("workout_adherence_pct") or adherence.get(
            "workouts_completed", 0
        ) / max(adherence.get("workouts_planned", 1), 1) * 100
        prompt += f"""
## Weekly Plan Adherence
- Meal plan followed: {meal_pct:.0f}% ({adherence.get('meals_followed', '?')}/{adherence.get('meals_planned', '?')} meals)
- Workouts completed: {workout_pct:.0f}% ({adherence.get('workouts_completed', '?')}/{adherence.get('workouts_planned', '?')} sessions)
"""

    # ── AI PROVIDER SELECTION ──────────────────────────────────────────────────
    try:
        from airflow.models import Variable
        provider = Variable.get("SYNTHESIS_AI_PROVIDER", default_var=_DEFAULT_AI_PROVIDER)
    except Exception:
        provider = _DEFAULT_AI_PROVIDER

    system_prompt = """You are a thoughtful, honest weekly observer — not a therapist, not a motivational coach.
You receive only numeric scores and habit/workout metrics. You never have access to the user's written journal text.

Your job: identify 2-3 genuine patterns, make 1-2 specific behavioral suggestions, and note any correlations between metrics.

Rules:
- Be concrete and behavioral. "You trained legs 3 times this week but logged brutal fatigue afterward" is good.
- Do NOT say "you should work on your stress levels" or other emotional interpretations — you can't see the text.
- Acknowledge when data is sparse (e.g. only 2 days logged) — don't fabricate patterns from thin data.
- Keep the output under 400 words.
- No bullet point lists of generic advice. No "Great job!" cheerleading.
- Write as one thoughtful friend summarizing observations to another. Honest, direct, useful.
- If mood, energy, and workout volume are high, say so plainly. If they're dropping, point it out.
- Make suggestions like: "Try adding X on Y" not "You should try to feel better about Z."
- End with one open question the person might find useful to reflect on."""

    synthesis_text = None
    model_used = None

    if provider == "ollama":
        import requests as req
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            resp = req.post(
                _OLLAMA_ENDPOINT,
                json={"model": _OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            synthesis_text = resp.json().get("response", "").strip()
            model_used = f"ollama/{_OLLAMA_MODEL}"
            log.info("Synthesis generated via Ollama (%s)", _OLLAMA_MODEL)
        except Exception as exc:
            log.error("Ollama synthesis failed: %s — falling back to Gemini", exc)
            provider = "gemini"

    if provider == "gemini" and synthesis_text is None:
        from agents.blog_agents import _gemini_flash
        try:
            synthesis_text = _gemini_flash(system_prompt, prompt)
            model_used = _GEMINI_MODEL_LABEL
            log.info("Synthesis generated via Gemini 2.5 Flash")
        except Exception as exc:
            log.error("Gemini synthesis failed: %s", exc)
            raise

    if not synthesis_text:
        raise RuntimeError("All AI providers failed to generate a synthesis")

    context['ti'].xcom_push(key='synthesis_text', value=synthesis_text)
    context['ti'].xcom_push(key='model_used', value=model_used)


# ── TASK 3: Save synthesis ─────────────────────────────────────────────────────

def task_save_synthesis(**context):
    """
    Upserts the synthesis into weekly_syntheses.
    Skips gracefully if no synthesis was generated (empty week).
    """
    from dag_db import execute, fetch_one

    synthesis_text = context['ti'].xcom_pull(key='synthesis_text', task_ids='generate_synthesis')
    model_used = context['ti'].xcom_pull(key='model_used', task_ids='generate_synthesis')
    structured_data = context['ti'].xcom_pull(key='structured_data', task_ids='collect_data')

    if not synthesis_text:
        log.info("No synthesis text — nothing to save")
        return

    import json
    week_start = structured_data['week_start']
    week_end = structured_data['week_end']
    avg_mood = structured_data['avg_mood']
    avg_energy = structured_data['avg_energy']
    habits_rate = structured_data.get('habits_completion_rate')
    workout_count = structured_data.get('workout_count')
    data_sources = structured_data.get('data_sources', ['journal_entries'])
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    existing = fetch_one(
        "SELECT id FROM weekly_syntheses WHERE week_start_date = %s",
        (week_start,)
    )

    if existing:
        execute(
            """UPDATE weekly_syntheses
               SET week_end_date = %s, avg_mood = %s, avg_energy = %s,
                   habits_completion_rate = %s, workout_count = %s,
                   synthesis_text = %s, data_sources = %s,
                   generated_at = %s, model_used = %s
               WHERE week_start_date = %s""",
            (week_end, avg_mood, avg_energy, habits_rate, workout_count,
             synthesis_text, json.dumps(data_sources), now, model_used, week_start)
        )
        log.info("Updated existing synthesis for week %s", week_start)
    else:
        execute(
            """INSERT INTO weekly_syntheses
               (week_start_date, week_end_date, avg_mood, avg_energy,
                habits_completion_rate, workout_count, synthesis_text,
                data_sources, generated_at, model_used)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (week_start, week_end, avg_mood, avg_energy, habits_rate, workout_count,
             synthesis_text, json.dumps(data_sources), now, model_used)
        )
        log.info("Inserted new synthesis for week %s", week_start)


# ── TASK 4: Lock old entries ───────────────────────────────────────────────────

def task_lock_old_entries(**context):
    """
    Sets is_locked=True on all journal_entries older than 24 hours.
    This is a cleanup pass — the router enforces locking independently.
    """
    from dag_db import execute

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        """UPDATE journal_entries
           SET is_locked = 1
           WHERE is_locked = 0
             AND created_at < DATE_SUB(NOW(), INTERVAL 24 HOUR)""",
        ()
    )
    log.info("Locked old journal entries (nightly cleanup pass)")


# ── DAG DEFINITION ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="life_os_weekly_synthesis",
    default_args=default_args,
    schedule_interval="0 23 * * 0",   
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "journal", "workouts"],
    doc_md="""
## Weekly Synthesis DAG

Runs every Sunday at 11 PM. Analyzes the week's mood/energy scores
and generates an AI pattern summary using NUMERIC DATA ONLY.

**Privacy**: journal text (content, gratitude, challenges) is NEVER
read or forwarded by this DAG. The AI sees only integers and counts.

**AI Provider**: Gemini 2.5 Flash (default) or local Ollama.
Set Airflow Variable `SYNTHESIS_AI_PROVIDER=ollama` to use local AI.
    """,
) as dag:

    collect = PythonOperator(
        task_id="collect_data",
        python_callable=task_collect_data,
    )

    generate = PythonOperator(
        task_id="generate_synthesis",
        python_callable=task_generate_synthesis,
    )

    save = PythonOperator(
        task_id="save_synthesis",
        python_callable=task_save_synthesis,
    )

    lock = PythonOperator(
        task_id="lock_old_entries",
        python_callable=task_lock_old_entries,
    )

    collect >> generate >> save >> lock