# MLB DFS Tools

Welcome! This repository holds everything you need to go from raw MLB data to playable DFS lineups, tournament simulations, and an optional web dashboard. The guide below is deliberately hand-holding—follow it from top to bottom and you will get working CSVs even if you have never touched this project before.

## What you get
- **Projection pipeline** – downloads historical stats, trains simple models, and produces fresh projections from a DraftKings slate export.
- **Lineup optimizer** – builds DraftKings, FanDuel, or IKB lineups that respect your stacking rules and ownership caps.
- **GPP simulator** – plays out tournaments thousands of times so you can see how often your lineups win or duplicate.
- **Dashboard + API** – a FastAPI service and React frontend that read the CSV outputs and turn them into visuals.

Keep reading for the exact commands to run each piece.

---

## 0. Before you start (one-time prerequisites)
You only have to do this once per machine.
- **Python 3.11 or newer** (check with `python --version`).
- **pip** (ships with Python) and the ability to create virtual environments (`python -m venv`).
- **Git** (so you can clone the repo).
- **Node.js 18+** if you want to run the optional dashboard (skip otherwise).
- **DraftKings / FanDuel account** so you can export the daily slate CSV.
- **Internet access** for the projection pipeline (it pulls data from public baseball APIs).

> Tip: On Windows the easiest route is through Windows Subsystem for Linux (WSL). All commands in this README assume a Unix-style terminal.

---

## 1. One-time project setup
Run these commands in a terminal, step by step.

1. **Clone the repo and open it**
   ```bash
   git clone https://github.com/your-name/MLB-DFS-Tools.git
   cd MLB-DFS-Tools
   ```

2. **Create a virtual environment** (keeps dependencies isolated)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows PowerShell use: .venv\\Scripts\\Activate.ps1
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # Optimizer & simulator extras (install them now to avoid surprises)
   pip install pulp tqdm numba seaborn matplotlib scipy
   ```

4. **(Optional) Install the dashboard frontend packages**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Create the folders the tools expect** (they may already exist)
   ```bash
   mkdir -p dk_data fd_data ikb_data output pipeline_artifacts
   ```

At this point the project is ready to accept data.

---

## 2. Create `config.json`
The optimizer and simulator read settings from `config.json` sitting in the repo root. Start from the sample and tweak it.

```bash
cp sample.config.json config.json
```

Open `config.json` in a text editor and review the fields. If your copy does not have a `global_team_limit` entry, add one (for example `"global_team_limit": 5`).

Here is what each field means:

| Setting | What it controls | Common value |
| --- | --- | --- |
| `projection_path` | Name of the projections file stored inside `dk_data/`, `fd_data/`, or `ikb_data/`. | `"projections.csv"`
| `ownership_path` | Name of the projected ownership file (optional but useful for sims). | `"ownership.csv"`
| `player_path` | Player ID file exported from the site (same folder as projections). | `"player_ids.csv"`
| `boom_bust_path` | Optional boom/bust data. Leave as-is if you do not use it. | `"boom_bust.csv"`
| `contest_structure_path` | Contest payout table used by the simulator (`cid` mode). | `"contest_structure.csv"`
| `team_stacks_path` | Stack ownership seed for the simulator. | `"team_stacks.csv"`
| `projection_minimum` | Lowest projection that will be kept (points). | `1`
| `randomness` | Percent of randomness injected into projections when building lineups. | `100`
| `at_least` / `at_most` | Optional dictionaries for rules like forcing or limiting players/teams. | `{}`
| `global_team_limit` | Maximum hitters from a single team in any lineup (add this field!). | `5` for DK, `4` for FD
| `primary_stack_min` / `max` | Size of your main stack (e.g. 4-5 if you want 4- or 5-man stacks). | `4` / `5`
| `secondary_stack_min` / `max` | Size of the secondary stack. | `3` / `4`
| `primary_stack_teams` | Comma list of teams you want as your primary stacks. `*` = any team. | `"NYY,ATL,SD,TEX"`
| `secondary_stack_teams` | Teams allowed in secondary stacks. | `"*"`
| `min_lineup_salary` | Lowest total salary allowed. | `49200`
| `max_pct_off_optimal` | How far sims allow a lineup to fall below optimal score (as a percentage). | `0.2` (20%)
| `pct_field_using_stacks` | % of the simulated field that stacks (simulator). | `0.75`
| `default_hitter_var` / `default_pitcher_var` | Used when a player has no provided standard deviation. | `0.5` / `0.3`
| `pct_max_stack_len` | How often the field uses the maximum stack length. | `0.65`
| `num_hitters_vs_pitcher` | Allowed hitters vs. the opposing pitcher. | `0`
| `pct_field_using_secondary_stacks` | Portion of field using a secondary stack. | `0.7`

Change anything you like; keep the file in valid JSON format (double quotes, commas between fields).

---

## 3. Gather slate data
You have two ways to get the projections the optimizer needs:

### Option A: Let the pipeline create projections (recommended)
1. **Train models (first run only).** This downloads multi-season data and can take several minutes.
   ```bash
   python -m pipeline.cli train --data-dir pipeline_artifacts
   ```
2. **Download the day’s slate from DraftKings.** Log in, open any MLB contest, click “Export to CSV”, and save the file (for example `~/Downloads/DKSalaries.csv`).
3. **Generate projections for that slate.** Replace the date with the slate date.
   ```bash
   python -m pipeline.cli project \
       --data-dir pipeline_artifacts \
       --slate ~/Downloads/DKSalaries.csv \
       --date 2024-04-01 \
       --output output/dk_projections_2024-04-01.csv \
       --template-output dk_data/projections.csv
   ```
   - `output/...` holds a full projection table with extra columns (vegas info, model type, etc.).
   - `dk_data/projections.csv` is trimmed to the format the optimizer expects. You can point `--template-output` to `fd_data/projections.csv` if you are working on FanDuel (just change the site later when you run the optimizer).
4. **(Optional) Inspect the projections.** Open the CSV in Excel/Numbers or run `head dk_data/projections.csv` to make sure it looks sensible.

### Option B: Drop in your own spreadsheets
Place the files the tools expect inside the site-specific folder.

| File | Where to save it | Minimum columns | Notes |
| --- | --- | --- | --- |
| Projections | `dk_data/projections.csv` (or `fd_data/...`) | `Name,Team,Pos,Fpts,Salary,Own%,Ord` plus optional `StdDev` | `Ord` is batting order (1–9). If `StdDev` is missing the simulator will estimate one using `default_*_var`.
| Player IDs | `dk_data/player_ids.csv` | `Name,ID,Game Info` (DraftKings) or for FanDuel use `Nickname` instead of `Name` | Exported from the site alongside salaries.
| Ownership (optional) | `dk_data/ownership.csv` | `Name,Own%` | Helps the simulator seed the market; leave blank if unknown.
| Team stacks (sim) | `dk_data/team_stacks.csv` | `Team,Own%` | One row per team with estimated stack ownership.
| Contest structure (sim) | `dk_data/contest_structure.csv` | `Place,Payout` or `Rank,Prize` | Required only when you run the simulator in `cid` mode.

Keep the headers spelled exactly as shown—the scripts lowercase them automatically.

---

## 4. Build lineups with the optimizer
Run the optimizer via `src/main.py`. The pattern is:

```bash
python src/main.py <site> opto <num_lineups> <num_unique_players>
```

Examples:
- `python src/main.py dk opto 150 2` → DraftKings, 150 lineups, at least 2 unique players between consecutive lineups.
- `python src/main.py fd opto 20 1` → FanDuel, 20 lineups, allow repeats.

What happens:
- The script reads `config.json` and the CSVs from `<site>_data/`.
- It prints progress to the terminal (stack settings, players loaded, etc.).
- When it finishes you will see a message like `Output complete. Lineups saved to output/dk_optimal_lineups_2024-04-01_12-30-00.csv`.

Open the CSV in `output/` to view the lineups. Each row lists the player names/IDs plus salary, projection, and ownership metrics.

---

## 5. Run the GPP simulator (optional but powerful)
Use this after you have projections in place. Command structure:

```bash
python src/main.py <site> sim <field_size_or_cid> <iterations_or_file> [iterations_if_using_file]
```

Common scenarios:
- **Sim a contest by field size:** `python src/main.py dk sim 50000 2000`
  - Simulates a 50,000-entry field for 2,000 iterations.
- **Use a contest payout file:** place `contest_structure.csv` in `dk_data/`, then run `python src/main.py dk sim cid 2000` (uses the payout table and 2,000 iterations).
- **Load your own entry file:** `python src/main.py dk sim 50000 file 2000`
  - Creates field lineups from the CSV inside `output/` before simulating.

Outputs (all land in `output/`):
- `dk_gpp_sim_lineups_<field>_<iterations>.csv` – top simulated lineups with win %, top 10%, ownership product, and duplication counts.
- `dk_gpp_sim_player_exposure_<field>_<iterations>.csv` – how often each player appeared in the field, plus simulated ROI if contest data was provided.

If you see warnings about missing standard deviations, consider adding a `StdDev` column to your projections or adjusting `default_hitter_var` / `default_pitcher_var` in `config.json`.

---

## 6. Analyze or visualise the results (optional)
### 6.1 Ownership analysis helper
If you have a CSV with lineups and ownership percentages, run:
```bash
python -m pipeline.cli analyze --lineups path/to/lineups.csv --output output/lineup_metrics.csv
```
This prints the top lineups to the terminal and, if you pass `--output`, writes a CSV including cumulative ownership, ownership product, and an estimated duplication interval.

### 6.2 Dashboard API + frontend
1. **Start the API** (reads the latest CSVs and serves JSON)
   ```bash
   uvicorn src.dashboard_api:app --reload
   ```
   It defaults to `http://127.0.0.1:8000`. Leave this running in a terminal.
2. **Launch the React app** (in a second terminal)
   ```bash
   cd frontend
   npm run dev
   ```
   Vite will open the dashboard at `http://localhost:5173` and proxy API calls to the FastAPI server.
3. **Production build** (optional)
   ```bash
   npm run build
   npm run preview
   ```

---

## 7. How everything fits together
1. Export the slate from the site.
2. Run the projection pipeline (or drop in your projections).
3. Save the CSVs inside `<site>_data/`.
4. Run the optimizer to create candidate lineups.
5. (Optional) Simulate the tournament field to evaluate duplication risk and ROI.
6. (Optional) Use the dashboard or analysis scripts to digest the outputs.

Repeat steps 1–6 each slate you play.

---

## 8. Troubleshooting & FAQ
- **“ModuleNotFoundError: No module named 'pulp'”** – install the solver with `pip install pulp` inside the virtual environment.
- **The optimizer says it cannot find `config.json`** – make sure you copied `sample.config.json` to `config.json` in the repo root.
- **The simulator complains about missing files** – double-check that the CSVs live in `dk_data/` (or the appropriate site folder) and use the column headers listed above.
- **Projections look empty** – confirm the DraftKings export matches the slate date you passed to the pipeline. Re-run the `project` command with the right `--date`.
- **Slow or stuck during training** – the first `train` command downloads several seasons of data. Subsequent runs reuse the cached files in `pipeline_artifacts/`. Pass `--force-refresh` only when you truly need fresh data.
- **Want to reset outputs?** – you can delete files inside `output/` and rerun the commands; the scripts recreate the files automatically.

---

## 9. Need a quick checklist for game day?
1. Activate your virtualenv: `source .venv/bin/activate`.
2. Download the latest DraftKings/FanDuel CSV.
3. `python -m pipeline.cli project ... --template-output dk_data/projections.csv`
4. `python src/main.py dk opto 150 2`
5. `python src/main.py dk sim 50000 2000`
6. Review the new CSVs inside `output/` and make adjustments as needed.

Happy building and good luck in your contests!
