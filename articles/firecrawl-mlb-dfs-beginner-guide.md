# Quip Trippin' with Blaine

Quip Trippin' with Blaine  
Musings on Data Science, DFS, Tennis, Traveling, Pop Culture

---

## Using Firecrawl to Build an MLB DFS Data Pipeline: A Beginner's Guide

*Posted on Tue 11 February 2025 in DFS*

This beginner-friendly guide walks through a miniature MLB DFS pipeline that leans on Firecrawl for web scraping, Claude or Codex for post-processing, and a lightweight React frontend. The end result is a SaberSim-style experience powered by real-world inputs from FanGraphs, DraftKings, RotoWire, and public Vegas odds.

The walkthrough targets Windows users, favors copy-and-paste commands, and keeps the architecture simple enough for weekend projects while leaving room to grow.

### Tool Analogy (Baseball Style)

- **Firecrawl = Data Scout.** Firecrawl visits remote sites, renders pages (even with JavaScript), and returns cleaned Markdown or JSON that is ready for downstream processing.
- **Claude or Codex = Analyst.** Once the scout delivers raw intel, an analyst parses tables, extracts schema-driven JSON, or writes concise summaries for the app UI.
- **Render = Stadium.** Render hosts the production web app, keeps secrets safe, and runs 24/7 so the pipeline is always available for your users.
- **n8n = Manager.** The manager sets the lineup: scheduling nightly runs, chaining tasks, and firing off alerts if anything breaks.

The remainder of this article follows a linear path: bootstrapping a Windows-friendly Node project, scraping data, persisting it, modeling projections, shipping a Next.js UI, and automating the workflow.

## 1. Firecrawl in Plain English

Before touching code, remember the roles:

- Firecrawl grabs structured data from the web and converts messy HTML into LLM-friendly Markdown or JSON.
- AI models such as Claude or Codex convert that cleaned text into the exact fields we need: player names, stat columns, and injury blurbs.
- Render keeps the web app running, protects API keys, and supports team collaboration.
- n8n orchestrates recurring scrapes, notifications, and retries.

Understanding the division of labor keeps the project manageable as complexity grows.

## 2. Windows Setup and a Firecrawl "Hello World"

### 2.1 Install Prerequisites

1. Download and install the Node.js LTS build for Windows (https://nodejs.org). Accept the defaults; npm ships with Node.
2. Open PowerShell or Command Prompt and verify versions:

   ```powershell
   node --version
   npm --version
   ```

3. Create a working folder and initialize a Node project:

   ```powershell
   mkdir firecrawl-mlb
   cd firecrawl-mlb
   npm init -y
   ```

4. Install the Firecrawl SDK and supporting packages:

   ```powershell
   npm install @mendable/firecrawl-js dotenv
   ```

### 2.2 Store the Firecrawl API Key

1. Sign up at https://firecrawl.dev and create an API key.
2. Create a `.env` file in the project directory and add:

   ```
   FIRECRAWL_API_KEY=fc-REPLACE_WITH_YOUR_KEY
   ```

   Windows tip: use `ni .env` in PowerShell if File Explorer will not allow dotfiles.

### 2.3 Write and Run the First Script

Create `crawl-test.js` and paste the following script. It saves both Markdown output and a small JSON metadata file for Mike Trout's FanGraphs page.

```javascript
import Firecrawl from '@mendable/firecrawl-js';
import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

const apiKey = process.env.FIRECRAWL_API_KEY;

if (!apiKey) {
  console.error('No API key found. Set FIRECRAWL_API_KEY in .env');
  process.exit(1);
}

const firecrawl = new Firecrawl({ apiKey });
const url = 'https://www.fangraphs.com/players/mike-trout/10155/stats?position=OF';

try {
  const result = await firecrawl.scrape(url, { formats: ['markdown'] });

  fs.mkdirSync('./data', { recursive: true });
  fs.writeFileSync('./data/trout.md', result.data.markdown);

  const title = result.data.metadata?.title || 'Unknown Title';
  fs.writeFileSync(
    './data/trout.json',
    JSON.stringify({ url, title }, null, 2),
  );

  console.log('Saved Markdown and JSON to the data folder.');
} catch (error) {
  console.error('Scrape failed:', error);
}
```

Run the script with `node crawl-test.js`; confirm the `data` folder now holds `trout.md` and `trout.json`.

## 3. Storing the Data

Start simple and grow as the project matures.

### 3.1 JSON Files

- Save scraped results as individual `.json` or `.md` files.
- Easy to inspect and share but awkward for filtering or joining across data sets.

### 3.2 SQLite via `better-sqlite3`

- Install with `npm install better-sqlite3`.
- SQLite runs in-process and stores everything in a single `.db` file.
- Example setup:

  ```javascript
  import Database from 'better-sqlite3';

  const db = new Database('./data/mlb.db');

  db.exec(`
    CREATE TABLE IF NOT EXISTS players (
      id INTEGER PRIMARY KEY,
      dk_id INTEGER UNIQUE,
      name TEXT,
      team TEXT,
      position TEXT
    );

    CREATE TABLE IF NOT EXISTS projections (
      player_id INTEGER PRIMARY KEY,
      proj_points REAL,
      proj_ownership REAL,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(player_id) REFERENCES players(id)
    );
  `);
  ```

- Ideal for local development. Remember that Render's free web services have ephemeral disks, so plan to migrate before deploying.

### 3.3 Postgres on Render

- Provision a free Postgres instance in the Render dashboard.
- Render supplies a `DATABASE_URL` connection string; add it as an environment variable.
- Switch to a Postgres client such as `pg` or an ORM. The schema can mirror the SQLite layout.
- Persisting in Postgres ensures data survives deploys and supports concurrent readers.

## 4. Running Locally vs. Hosting on Render

- **Local CLI scripts** (like `crawl-test.js`) are ideal for development and debugging.
- **Server-side API routes** bring the scrape logic into your web app so it can run on Render. In a Next.js project, add `pages/api/crawl.js` or the `app/api` equivalent. Keep secrets server-side by reading `process.env`.
- Secure the route with a static token or auth guard if the app is public.

Example API handler:

```javascript
import Firecrawl from '@mendable/firecrawl-js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const firecrawl = new Firecrawl({ apiKey: process.env.FIRECRAWL_API_KEY });

  try {
    const result = await firecrawl.scrape(
      'https://www.fangraphs.com/players/mike-trout/10155/stats?position=OF',
      { formats: ['markdown'] },
    );

    return res.status(200).json({ success: true, data: result.data });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Crawl failed' });
  }
}
```

## 5. Gathering DFS Data Sources

### 5A. FanGraphs Stats

- Target leaderboard pages for batch scrapes (e.g., batting wOBA and PA, pitching K% and IP).
- Use Firecrawl Markdown exports and feed the table text to an AI prompt: "Extract an array of records with name, team, PA, wOBA." Export per season or per slate.

### 5B. DraftKings Salaries

- Download the slate CSV and parse each row into `{name, dk_id, salary, team, position}`.
- Regular expression example: `^(.+)\s+\((\d+)\)$` to split "Name (ID)".
- Normalize names (remove punctuation, accents, Jr./Sr. suffixes) to improve joins.

### 5C. RotoWire Injuries and Depth Charts

- Scrape the injury report page and convert to JSON with an AI prompt: "Output name, injury_status, injury_note." Apply codes such as ACT, DTD, IL10.
- Optional: parse depth charts or starting lineups for batting order context.

### 5D. Vegas Odds and Implied Totals

- Choose a public odds site (ScoresAndOdds, VegasInsider, etc.).
- Extract team matchups, moneylines, and game totals. Compute implied team totals or scrape them directly when available.
- Merge totals back into the player dataset by team abbreviation.

## 6. AI Post-Processing with Claude or Codex

### 6.1 Structured Extraction Prompts

Example prompt for FanGraphs batting data:

```
Extract a JSON array from the following table. Fields per row:
- player_name (string)
- team (string, 2-3 letter abbreviation)
- woba (number between 0 and 1)
- plate_appearances (integer)

Return JSON only.
```

### 6.2 Validation Tips

- Always run `JSON.parse` and catch errors.
- Strip leading ```json fences if the model includes them.
- Sanity check numeric ranges (e.g., wOBA between 0 and 1).

### 6.3 Optional Player Summaries

Reuse the same context to ask the model for 2 to 3 bullet points summarizing the player's recent form, injury news, or matchup outlook.

## 7. Toy Projection and Ownership Formulas

### 7.1 Hitter Projection Skeleton

```javascript
function projectHitter(player) {
  if (!player.woba) return 0;

  let plateAppearances = 4.2;
  if (player.team_total && player.team_total > 5) {
    plateAppearances = 4.5;
  }

  let projection = player.woba * 10 * plateAppearances;

  if (player.order) {
    projection *= 1 + (9 - player.order) * 0.02;
  }

  if (player.injury_status && player.injury_status !== 'ACT') {
    projection = 0;
  }

  return projection;
}
```

### 7.2 Pitcher Projection Skeleton

```javascript
function projectPitcher(player) {
  if (!player.k_rate) return 0;

  let innings = 6;
  if (player.opponent_total && player.opponent_total > 4.5) {
    innings = 5;
  }

  const strikeouts = player.k_rate * innings * 3;
  let projection = strikeouts * 2 + innings * 5;

  if (player.opponent_total) {
    projection -= player.opponent_total * 2;
  }

  if (player.team_total && player.opponent_total && player.team_total > player.opponent_total) {
    projection += 4;
  }

  if (player.injury_status && player.injury_status !== 'ACT') {
    projection = 0;
  }

  return projection;
}
```

### 7.3 Ownership Heuristics

- Sort hitters by projection. Assign top three around 20 percent, next five around 10 percent, and the rest between 2 and 5 percent.
- Sort pitchers by projection. First two can fall in the 35 to 50 percent range with the remainder trailing.
- These are placeholder numbers: refine them with historical contest data once available.

## 8. Building a Minimal Next.js App

### 8.1 Pages to Include

- **Home:** Buttons to trigger crawls, links to tables and lineup tools.
- **Players Table:** Server-side fetch from JSON, SQLite, or Postgres. Provide sorting by column.
- **Lineup Builder:** Allow selecting players, enforce the salary cap, and sum projected points.

### 8.2 Deployment to Render

1. Push the repo to GitHub.
2. Create a Render Web Service tied to the repo.
3. Set build command `npm install && npm run build` and start command `npm run start`.
4. Add environment variables: `FIRECRAWL_API_KEY`, `DATABASE_URL` (if applicable), AI provider keys, and any crawl secrets.
5. Deploy. Render sleeps free services after 15 minutes of inactivity but wakes on visit.

## 9. Scheduling with n8n

- Use a cron-style schedule node (for example, run daily at noon).
- Call the app's crawl endpoint with an HTTP node. Include a shared secret header to prevent unauthorized access.
- Optionally add Slack or email nodes for success/failure notifications.
- Interrupt long chains with retries and error branches where necessary.

Community members have published Firecrawl-specific n8n nodes, so you can also orchestrate Firecrawl directly from n8n without touching your app.

## 10. Web Scraping Etiquette

- Check each site's `robots.txt` before scraping.
- Respect crawl delays and rate limits; Firecrawl includes polite defaults but slow down further if you collect dozens of pages.
- Identify your user agent when feasible.
- Avoid redistributing proprietary data without permission.
- Implement error detection for CAPTCHAs or access denials and back off if encountered.

## 11. Troubleshooting and Next Steps

- Empty responses? Enable browser rendering or wait for `networkidle` in Firecrawl options.
- AI returning malformed JSON? Tighten prompts, request "JSON only," and trim code fences before parsing.
- Name collisions between data sources? Build a normalization helper and maintain a small alias dictionary for tricky cases.
- Render deployment issues? Inspect build logs, confirm environment variables, and pin the Node engine if dependencies require it.
- Scaling ideas: integrate official MLB APIs, add Monte Carlo simulations, inspect live ownership results, and introduce better UI/UX with charts or filters.

### Appendices

**Appendix A: DraftKings CSV Parsing Snippet**

```javascript
import fs from 'fs';

const raw = fs.readFileSync('DKSalaries.csv', 'utf-8').trim();
const lines = raw.split('\n');
const players = [];

for (let i = 1; i < lines.length; i += 1) {
  const cols = lines[i].split(',');
  const position = cols[0];
  const nameId = cols[1];
  const salary = parseInt(cols[2], 10);
  const team = cols[3];

  const match = nameId.match(/^(.+)\s+\((\d+)\)$/);
  if (!match) continue;

  const name = match[1];
  const dkId = parseInt(match[2], 10);

  const normalized = name
    .toLowerCase()
    .replace(/\./g, '')
    .replace(/ jr$/, '')
    .replace(/ iii$/, '')
    .replace(/'/g, '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim();

  players.push({ position, name, dk_id: dkId, team, salary, normalized });
}
```

**Appendix B: Sample Database Schema**

```sql
CREATE TABLE players (
  id SERIAL PRIMARY KEY,
  dk_id INTEGER UNIQUE,
  name TEXT NOT NULL,
  team TEXT,
  position TEXT
);

CREATE TABLE stats (
  player_id INTEGER REFERENCES players(id),
  woba REAL,
  k_rate REAL,
  plate_appearances INTEGER,
  innings_pitched REAL
);

CREATE TABLE projections (
  player_id INTEGER PRIMARY KEY REFERENCES players(id),
  proj_points REAL,
  proj_ownership REAL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Appendix C: Glossary**

- **wOBA:** Weighted on-base average; advanced stat measuring offensive contribution.
- **K%:** Strikeout percentage; for hitters it tracks whiffs, for pitchers it tracks strikeouts recorded.
- **PA:** Plate appearances; how many times a hitter comes to the plate.
- **IP:** Innings pitched; often shown with thirds (for example, 5.2 = 5 and two thirds).
- **Implied team total:** Expected runs for a team derived from betting lines.
- **Ownership:** Percentage of DFS lineups that include a given player.
- **Stack:** Roster cluster of hitters from the same team to maximize correlation.
- **DFS:** Daily fantasy sports; build new lineups each slate under a salary cap.
- **SaberSim-style:** Uses simulations to derive projection distributions and lineup optimizations.

### References

1. Firecrawl Quickstart: https://docs.firecrawl.dev/introduction
2. Node SDK for Firecrawl: https://docs.firecrawl.dev/sdks/node
3. Mendable Firecrawl MCP Server: https://github.com/firecrawl/firecrawl-mcp-server
4. Protect API keys with `.env`: https://medium.com/@oadaramola/a-pitfall-i-almost-fell-into-d1d3461b2fb8
5. FanGraphs Mike Trout stats: https://www.fangraphs.com/players/mike-trout/10155/stats?position=OF
6. SQLite is serverless: https://www.sqlite.org/serverless.html
7. Render free tier overview: https://render.com/docs/free
8. Next.js environment variable guide: https://nextjs.org/docs/pages/guides/environment-variables
9. Rotogrinders' analysis on implied totals: https://rotogrinders.com/articles/mlb-dfs-how-accurate-are-vegas-implied-totals-1967459
10. Robots.txt etiquette: https://www.promptcloud.com/blog/how-to-read-and-respect-robots-file

