# Smart Analytics

Pulls your Garmin Connect data into a local database and turns it into training
analytics with a GUI — muscle-level strength balance, running performance, split-level
run analysis, concurrent-training interference, and an AI coaching layer.

Local-first: everything is cached in a SQLite file on your machine. The only outbound
calls are to Garmin (to fetch) and, if you enable the coach, to the Claude API — which
receives *computed metrics only*, never raw activity data.

## What it answers

**Strength — which muscles are falling behind, and why**

If you train from a structured workout, Garmin's workout definition already states which
muscles it assigns to each exercise — so those assignments are synced and used first,
translated from Garmin's anatomical names (`BICEPS_FEMORIS`, `DELTOID_POSTERIOR`) into an
18-muscle model. A built-in table of 30 exercise categories and 44 named variants covers
anything Garmin didn't label. Either way credit is fractional — a barbell row gives the
upper back a full set and the biceps half of one — and every muscle is scored 0–100 on
four independent signals:

| Signal | What it measures |
|---|---|
| Volume | weekly effective sets against your target range |
| Trend | estimated-1RM slope for the exercises that load it |
| Recency | days since the muscle was last loaded |
| Balance | how it compares with its structural counterpart |

Plus antagonist ratios (push:pull, quads:hamstrings, front:rear delts), movement-pattern
coverage, per-exercise estimated 1RM trends with stall detection, and a report of any
exercises it couldn't map — so nothing is silently dropped.

**Running — is fitness actually improving, and where's the gain**

The headline metric is **aerobic efficiency** (metres per heartbeat), because pace alone
is confounded: a faster run at a higher heart rate isn't necessarily fitness. Alongside it:

- **Personal training paces** built from Garmin's own lactate-threshold estimate — so
  "you're stuck in the grey zone" becomes "your easy runs average 5:34/km; your easy
  range is 5:57–6:52/km, about 22 s/km too quick"
- **Aerobic decoupling** from per-lap data — whether heart rate drifts upward at constant
  pace through a long run. A run averaging 150 bpm might hold 145 throughout or climb
  from 138 to 162; same average, opposite meaning
- **Interval execution** — whether reps hold pace or fade, which tells you if the session
  started too hard
- Intensity distribution, consistency, Riegel-normalised personal bests and race
  predictions, negative-split rate, cadence

**Load, recovery and concurrent training**

Every activity type on one scale (Garmin's own training load where available, Banister
TRIMP estimated from heart rate where not), giving ACWR, fitness/fatigue/form, and Foster
monotony. Plus resting HR, HRV, sleep and body battery against baseline.

For hybrid athletes, the interference analysis looks at the *calendar* rather than the
totals: leg sessions colliding with quality runs, same-day session order, whether hard
days cluster or spread, and how load splits between disciplines.

**What to do next**

A rule-based recommendation applied in a fixed order — open niggles, then recovery state,
then interference from yesterday, then the biggest gap — with every reason shown so you
can overrule it. Plus countable weekly targets and a shareable digest export.

**Progress over time**

Each sync stores a snapshot, so the diagnostics themselves can be trended: "hamstrings
went from 68 to 41 over six weeks, because weekly volume went from 2.4 to 9.1 sets."

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then edit it
streamlit run app.py
```

Open <http://localhost:8501>.

### Try it without a Garmin account

Turn on **Demo mode** in Sync & Settings and hit Run sync. It generates a realistic
400-day history — with deliberate weaknesses planted so you can see what the analytics
find — and runs it through the same normalisation and storage path as a real sync.

### Connecting Garmin

Sign in from the app: **Sync & settings → Garmin Connect account**, enter your email and
password, and if Garmin asks for a multi-factor code you'll get a second box to enter it.
Or set `GARMIN_EMAIL` and `GARMIN_PASSWORD` in `.env` if you'd rather not type them each
time.

Either way the password is used **once**, to mint OAuth tokens, which are cached in
`.garmin_tokens/` (gitignored). Later syncs resume from those, so MFA is prompted at most
once per token lifetime. The password is never written to the database, and **Sign out**
deletes the cached tokens.

**If a verification code never arrives:** the login prefers Garmin's browser sign-in form
over the mobile API login, on the theory that only the former dispatches a code. That
theory is unproven — `garth`, the reference implementation, verifies codes without ever
requesting one, which suggests Garmin dispatches on the login itself and the flow doesn't
matter. The preference is kept because it mirrors what a browser does, not because it's
known to fix anything.

What is known: if your account uses an authenticator app, nothing is emailed at all and
the prompt says so; and Garmin throttles repeated MFA attempts, so a burst of retries can
stop codes arriving. The panel's **"The code hasn't arrived"** section reports which flow
raised the challenge and whether the browser form was actually used, which is the evidence
worth having before theorising further.

Garmin has no public consumer API — the official Developer programme is partner-only — so
this uses [`garminconnect`](https://github.com/cyberjunky/python-garminconnect), which
speaks the same OAuth flow as the mobile app. Two consequences worth knowing:

- **Rate limits are real.** Detail endpoints cost one request per activity, so sync is
  bounded: `detail_batch` strength workouts and `split_batch` runs per run. A long
  history fills in over a few syncs rather than in one hit.
- **Payload shapes vary** by device generation and account age. Every physiological
  metric is optional; a missing one shows as unavailable rather than crashing or being
  guessed at.

### The AI coach

Set `ANTHROPIC_API_KEY` in `.env`. Without it everything still works — you get the
rule-based summary instead of the narrative layer.

The design rule is that **Claude never computes numbers**. The analytics engines do all
the arithmetic and hand over a compact JSON briefing; Claude's job is to interpret it —
rank what matters, explain the mechanism, turn findings into a plan. You can inspect the
exact briefing on the AI Coach page. If a number isn't in there, the model has no way to
know it.

## CLI

```bash
smart-analytics sync --demo        # load generated data
smart-analytics sync              # pull from Garmin (add --incremental to resume)
smart-analytics report            # print findings to the terminal
smart-analytics report --json b.json   # also dump the AI briefing
smart-analytics ui                # launch the GUI
```

## Layout

```
app.py                          Streamlit launcher
src/smart_analytics/
  config.py                     settings from .env
  db.py                         SQLite schema, upserts, migrations
  reporting.py                  weekly digest (Markdown + HTML)
  cli.py                        command line interface
  garmin/
    client.py                   Garmin Connect client + normalisation
    physiology.py               threshold, zones, readiness, splits, records
    workouts.py                 structured workouts, Garmin's muscle assignments
    sync.py                     bounded incremental sync
    sample.py                   demo data with planted weaknesses
  domain/
    muscles.py                  18-muscle taxonomy, balance pairs
    exercises.py                exercise → muscle resolution (Garmin first)
    garmin_muscles.py           Garmin's anatomical names → the 18-muscle model
  analytics/
    strength.py                 volume, e1RM trends, lag detection
    running.py                  pace, efficiency, bests, predictions
    zones.py                    pace zones from Garmin's threshold
    splits.py                   decoupling, interval quality, pacing
    load.py                     TRIMP, ACWR, fitness/fatigue, recovery
    hybrid.py                   concurrent-training interference
    niggles.py                  niggle log against load history
    snapshots.py                progress over time
    prescription.py             what to train next
    report.py                   assembles everything; builds the AI briefing
  ai/insights.py                Claude coaching layer
  viz/                          validated palette + chart builders
  app/                          Streamlit pages
tests/                          80 tests
```

## Notes on method

Choices that affect how the numbers should be read:

- **Effective sets, not tonnage**, as the primary strength unit. Tonnage is dominated by
  whichever lift moves the most absolute weight and silently drops bodyweight work;
  fractional-credit set counts track the stimulus each muscle actually receives. Tonnage
  is still reported, since it's the right unit for one lift over time.
- **Epley for estimated 1RM**, with reps capped at 15 — the formula degrades badly in
  high-rep territory, so a 30-rep set can't imply a 2× 1RM.
- **Threshold-anchored pace zones**, not percentage-of-max-HR. Threshold pace is the most
  reproducible field anchor there is; max HR depends on a max you probably haven't measured.
- **Trends are only reported when they're supportable** — at least 4 sessions spanning
  21 days, or the metric reads "insufficient data" rather than fitting a line to noise.
- **Decoupling is computed on metres-per-heartbeat**, not raw heart rate, so a run that
  slows down as well as drifting isn't scored as if it held pace.
- **Niggle correlations are associations, not causes.** n=1 with confounders everywhere.
  The app shows what the load was doing around each onset and stops there. It doesn't
  diagnose, and it points persistent or severe entries at a professional.

## Tests

```bash
pytest
```

The suite runs the demo client through the real sync and analytics path, so it covers
production code rather than a test-only shortcut.

## Limitations

- Garmin's per-set data only exists for strength workouts recorded on a watch with
  exercise tracking on. Manually logged sessions carry no set data, so they can't be
  included in the muscle model.
- No per-side data, so left/right imbalance can't be assessed.
- Garmin's muscle assignments only cover exercises that appear in a structured workout (or
  in the exercise library, where an account exposes one). Anything else falls back to the
  built-in table, which isn't exhaustive — unmapped exercises are listed on the Strength
  page; add them to `NAME_PROFILES` in `domain/exercises.py`.
- Garmin sometimes labels a muscle too coarsely to use — an unqualified `DELTOID` can't be
  split into front/side/rear, which are separate balance targets — so those names are
  reported on the Sync page rather than guessed at.
- Not medical advice.
