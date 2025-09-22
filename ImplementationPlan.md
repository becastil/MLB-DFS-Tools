# Implementation Plan for MLB DFS Projection & Ownership Pipeline

## Overview

This plan outlines a complete pipeline to generate daily fantasy baseball player projections and ownership estimates using three seasons of data. The approach integrates data from FanGraphs (player stats), DraftKings (slate info), and RotoWire (lineups), coupled with a custom machine learning model (for example, linear regression) to project fantasy points, and a method to estimate player ownership per slate. Below, we detail each component of the pipeline and discuss feasibility and implementation steps.

## Data Collection and Sources

### 1. FanGraphs Statistical Data

FanGraphs offers extensive historical player data but no official public API (angelineprotacio.com). We will scrape data from FanGraphs using Python (for example, requests plus BeautifulSoup) since FanGraphs pages are mostly static HTML (no heavy JavaScript) (angelineprotacio.com). This makes it feasible to retrieve stats without a browser emulator. Key data to collect includes:

- Historical Game Logs: For each player, gather game-by-game performance data for the last three seasons. FanGraphs provides game logs by player and year (for example via player pages or using player IDs) (billpetti.github.io). These logs contain detailed per-game stats (hits, HR, RBI, strikeouts, and so on), which can be used to calculate fantasy points for each game. Using an existing tool like the pybaseball library can simplify this because it automatically scrapes FanGraphs and Baseball Reference for historical stats (github.com). We can leverage such libraries to avoid building every scraper from scratch.
- Advanced Metrics: In addition to raw game stats, FanGraphs has advanced metrics (wOBA, wRC+, K%, ERA, and others) that can serve as features in our model. We can scrape relevant leaderboards or player pages for splits (for example, a batter's stats versus left-handed pitchers) if needed. It is important to collect both hitter and pitcher data, as we will likely build separate models for each.
- Data Volume: Three seasons of data (for example, 2022 through 2024) means on the order of tens of thousands of player-game entries, which is a manageable size for scraping and modeling. We will implement the scraping with care (rate limiting, small batches) to avoid being blocked (angelineprotacio.com). Optionally, using FanGraphs' provided "Export Data" buttons manually for large tables or employing retry logic in code can improve reliability.
- Storage: After scraping, store the data in a structured format (CSV or database). For instance, maintain separate tables for hitters and pitchers game logs with fields for player ID, date, opponent, stats, and similar fields. We will also capture Fantasy Points as a derived field by applying DraftKings scoring rules to each game's stats (for example, 3 points per single, 5 per double, and so on, including pitcher scoring). This labeled data (game to actual fantasy points) will be the training target for our machine learning model.

### 2. DraftKings Player Pool Data

DraftKings provides a CSV download of each day's slate player list (names, positions, teams, salaries, and an internal player ID) (reddit.com). We will integrate this by:

- Automating CSV Retrieval: Each slate (contest) on DraftKings has an "Export to CSV" option on the lineup screen (reddit.com). If possible, we can programmatically fetch this (for example, by constructing the URL or using the DraftKings API if available). Otherwise, a manual step or a headless browser script may be needed to simulate clicking the export. The CSV has entries like "Jose Altuve (123456)" as the name field, where 123456 is DraftKings' player ID.
- Parsing and Matching: We will parse the player name and ID from this CSV. The name will be used to match the player to our projection data. Since name formats might differ slightly between sources (for example, "Ronald Acuna Jr" versus "Ronald Acuna Jr." with punctuation), we might use a simple name normalization (remove punctuation, Jr or Sr) and possibly maintain a mapping of known discrepancies. In many cases, joining on name should be sufficient if our FanGraphs data includes player names consistently. If FanGraphs data is keyed by player ID (their own ID), we may need a one-time mapping of FanGraphs IDs to names as well.
- Using the Player ID: The DraftKings ID from the CSV will be important for output because many optimizers require that ID to identify players. We will ensure that we carry this ID through in our final output alongside the name, so the optimizer or simulator knows exactly which player the projections belong to.
- Slate-Specific Info: The DraftKings CSV also contains each player's position and salary for that slate. We will not alter those, but they are crucial context for modeling ownership (for example, salary and position influence ownership). We will keep this data accessible for the ownership model.

### 3. RotoWire Lineup Data

Daily starting lineups (who is actually in the lineup and batting order) are critical for refining projections. RotoWire's MLB daily lineups page is a reliable source updated throughout the day (rotowire.com). Our plan for lineup integration:

- Accessing Lineups: RotoWire lists each game with confirmed and projected lineups, typically including batting order, starting pitchers, and similar info. We can scrape this page periodically (for example, every five to ten minutes near lineup lock) or use an API. In fact, developers have created APIs that parse RotoWire's lineup page and output JSON (mattgorb.github.io), indicating the feasibility of programmatic access. If budget allows, RotoWire's own API for subscribers could be used for more robust data. Otherwise, we will scrape the HTML (using requests and BeautifulSoup or an unofficial JSON feed) to get the starting players and their batting order for each team.
- Using Lineup Info: Once we have confirmed lineups, we will adjust our projections by excluding non-starters (if a player in the DraftKings pool is not in the starting lineup their projection should be essentially zero, since they probably will not play enough to accumulate points), applying batting order adjustments (incorporate batting order as a feature or post-process adjustment), and handling starting pitchers appropriately.
- Timing: Lineup integration means our pipeline likely runs in two phases: an initial projection run (using expected lineups or last-known info), and an update closer to lock when most lineups are confirmed. Given the importance of final lineups, automation (scraping RotoWire's page) is feasible to incorporate in real time (mattgorb.github.io).

## Machine Learning Pipeline for Projections

### 4. Feature Engineering for Projections

With historical data from FanGraphs and daily context data, we will construct features to feed into a predictive model of fantasy points. Key features for each player-game (slate) could include:

- Player Recent Performance: A rolling average of the player's fantasy points or key stats (for example, last 14 days, last 30 days) to capture hot and cold streaks.
- Player Long-Term Averages: Overall skill indicators such as the player's season wOBA, OPS (for hitters) or ERA, K% (for pitchers), possibly aggregated from the past seasons. These give a baseline expectation of talent level.
- Platoon Splits: If available, use stats versus left-hand or right-hand pitching as appropriate. For example, if a hitter is facing a lefty pitcher today, include the hitter's historical performance versus lefties (from FanGraphs splits) as a feature. Similarly for pitchers versus left or right batters.
- Matchup Variables: Features describing the opponent: for a hitter, the opposing pitcher's metrics (for example, opponent pitcher's WHIP, K per 9, HR per 9, or an overall quality indicator like opponent's fantasy points allowed on average). For a pitcher, features about the opposing offense (team batting stats like average runs scored, team K% which affects pitcher strikeouts). These can be derived from the last three seasons of data or current season aggregates.
- Park Factor: The ballpark can greatly influence scoring. We will incorporate a park factor (for example, FanGraphs park factor for run scoring or home runs for that stadium) (billpetti.github.io) as a feature. Games at Coors Field get a boost to hitters' projections, while a pitcher there would be downgraded.
- Vegas Odds (Optional): If accessible, Vegas lines for game total and team implied runs can serve as an excellent feature for expected scoring environment. Since the user is open to any approach (and even paid data), we might integrate an implied run total for each player's team and the moneyline win probability for pitchers (which correlates with chances for a win and thus extra points). This can significantly enhance accuracy, though it requires pulling odds from an API or site (an optional enhancement if time permits).

We will assemble these features for each player in each slate. Much of this can be prepared once lineups and matchups are known (for example, after scraping lineups, we know which pitcher each hitter faces and so on to choose the correct features).

### 5. Model Selection and Training

We plan to start with a linear regression model to predict a player's fantasy points for a given game, using the features above. Linear regression is a good baseline due to its simplicity and interpretability, and it has been used successfully in sports modeling (reddit.com). Implementation steps:

- Dataset Preparation: Using the collected three-season game logs, create a training dataset where each record is a historical game for a player, with the features as described (player's stats pre-game, opponent info, and so on) and the target is the actual DraftKings fantasy points scored in that game (computed from the stats). We should split data by year or date (for example, use 2.5 seasons for training, and the most recent half-season for validation or testing) to evaluate out-of-sample performance.
- Train a Regression Model: Fit an ordinary least squares linear regression (or a regularized version like ridge regression if necessary to avoid overfitting on correlated stats). We will likely train separate models for hitters and pitchers, since their stat profiles and scoring are different.
- Model Evaluation: Evaluate the model on validation data (for example, last season's games) with metrics like R squared and RMSE to ensure it captures variance in fantasy points. If linear regression proves too limited we can consider more complex models such as decision tree ensembles (random forest or XGBoost) which can capture interactions. However, given the data volume and need for interpretability, starting with linear or ridge regression is reasonable.
- Iterating Features: We will refine features based on results. For example, if the model underestimates players in elite hitting environments, we might incorporate an interaction between park factor and player power. We will also ensure the model is not biased by outdated data (hence using the most recent seasons and possibly weighting recent games higher).
- Prediction Generation: Once the model is satisfactory, on any given slate day we will generate projections by feeding in the current slate's features (player's stats up to today, today's matchup info, and so on). This yields a projected DraftKings fantasy point value for each player. These projections are the core output to integrate into the CSV.

Feasibility: This machine learning approach is feasible with open data and tools. Python's scikit-learn library can quickly train linear models on thousands of samples. Three seasons of data gives a robust sample size for general trends, though baseball has high variance day-to-day. Linear regression can serve as a baseline and we can layer in domain knowledge to improve it.

## Ownership Projection Methodology

### 6. Projected Ownership Strategy

Estimating each player's DFS ownership percentage is challenging but we will approach it with a mix of data-driven heuristics and potential modeling:

- Factors Influencing Ownership: Ownership is influenced by salary value (point per dollar), player popularity, recent performance, and slate context (for example, a cheap backup becoming starter). Because of this complexity, purely modeling ownership from scratch may be less reliable. However, we can design a simplified model using key predictors.
- Our Projection and Salary: Compute each player's value ratio (projection divided by salary). Players with the highest point per dollar in each position tend to attract heavy ownership, as they are seen as good values. We can rank players by this metric.
- Opportunity and Role Changes: Identify players who suddenly have increased opportunity (for example, moved to top of lineup or replacing an injured starter at low salary). We will flag such cases.
- Star Players: Elite players (high salary, high projection) will still garner ownership, though usually proportional to how easily they fit into lineups. Our model can consider how many high-priced players have good projections on the slate.
- Positional Depth: If a position has few good options, the top one or two at that position will naturally have higher ownership. Conversely, if a position has many similar alternatives, ownership may spread out.
- Recent Performance and News: Players on a hot streak or who have media buzz tend to be picked by casual players (reddit.com). If feasible, we will include a recent fantasy point average or last-game points as a proxy. Also, last-minute news can shift ownership but is hard to quantify; we might reasonably ignore such niche factors in an automated model.

Heuristic or Rule-Based Estimation: Given the above factors, we can create a formula or set of rules to assign a preliminary ownership percentage. For example, assign a baseline ownership to each player based on value ranking (scale top value pitcher close to 40 percent in large tournaments, and so on). Then adjust by boosting ownership for popular names and adjusting for chalk situations (if a minimum-priced hitter projects nearly as well as expensive peers, expect a big ownership spike). This rule-based approach leverages domain knowledge more than pure data fitting, which might be necessary due to limited explicit ownership data.

Data-Driven Modeling (if data available): If we can obtain historical ownership percentages (for example, by scraping contest results or using any provided public info), we could train a regression model for ownership. The model's features would be those listed above, and target the actual ownership percentage. However, getting a large dataset of past ownership requires saving contest info daily. Initially, we may rely on educated estimates.

Simulation-Based Approach: Another feasible technique is to use our optimizer or simulation tool itself to derive ownership. We can simulate many lineups by optimizing for projection (as many DFS players or optimizers would) and see how frequently each player appears. This mimics the field using projections to build lineups (reddit.com). For example, running 1,000 optimal lineup iterations and counting player frequency can yield a pseudo projected ownership. This approach would directly utilize our projections and give higher counts to high-value players.

Finalize Ownership Estimates: The output will be each player's projected ownership percentage for the slate. We will likely express this as a percentage (0 to 100). It is wise to double-check that the total ownership across players and positions makes sense (for example, ensure we are not grossly over or understating chalk versus punts).

Feasibility: Projecting ownership from scratch is possible but difficult (reddit.com). The accuracy will not match industry experts immediately, because it is influenced by collective human behavior and information flow. However, for feasibility, a heuristic model can be implemented quickly and improved over time. Another angle is to consider subscribing to an existing DFS data service that provides ownership projections, and incorporate those directly. This could be a straightforward solution (the user is open to paid solutions). In summary, we propose starting with our own ownership estimator and evaluating its accuracy; if it is insufficient and time permits, explore external data as a supplement.

## Integration and Output

### 7. Combining Projections with DraftKings CSV

Once we have the projections and ownership for all players on a slate, we will integrate them into the DraftKings player CSV. Implementation steps:

- Join Data: Using the player name (or a unique identifier if we established one), join our projection and ownership values to the DraftKings player list. We will ensure every player in the DraftKings CSV gets a projection. There will be cases like a minor-league call-up or recent debut where our three-season model has little data. For those, we may assign a conservative projection.
- Populate CSV Columns: We will add new columns for projected fantasy points and projected ownership. The rest of the columns from the original CSV remain as-is. If the optimizer or simulation tool expects a specific format, we will adhere to that.
- Quality Check: We will implement checks, for example, if a player is not in the lineup their projection should be near zero. Verify that extremely high projections or ownerships make sense.

### 8. Optimizer and Simulation Tool Integration

The final goal is for these projections to feed into an existing lineup optimizer and a simulation tool. Integration will depend on how those tools ingest data:

- If the optimizer is a script or software that reads a CSV, our job is done by providing the CSV with the added columns.
- If the simulation tool is code-based (for example, a Python or R script that is part of a repository), we can package our pipeline as a module or set of functions that the tool can call.
- We will ensure that the output adheres to any schema required. For example, if the simulation expects a JSON or a database input instead of CSV, we will adapt accordingly.

Automating the Workflow: To make this pipeline run daily per slate, we can automate the sequence. In the morning run the data fetch, a few hours before games incorporate lineup data and adjust projections, and optionally re-run just before lock to catch any late lineup swaps or scratches. This could be orchestrated with a scheduler (cron job) or a simple script that sleeps and checks for updates. Because the user said scheduling is not needed now, we will initially focus on building the pipeline pieces that can be run manually in sequence.

### 9. Testing and Validation

Before fully relying on this system, we will test it on a few past slates: feed in historical slates (with known lineups) and see how our pipeline performs (both technically, and how close projections were to actual results or how reasonable ownership guesses were). This will highlight any issues in data matching or model predictions that we can refine.

## Feasibility and Considerations

Overall, the approach is feasible with moderate effort. All required data is available from public sites or via lightly paywalled APIs, and there are known methods to obtain it (scraping FanGraphs (angelineprotacio.com), using RotoWire for lineups (mattgorb.github.io), exporting DraftKings data (reddit.com)). The machine learning component (linear regression and similar models) can be implemented using open-source libraries and should run efficiently on the dataset size at hand.

### Challenges and Mitigations

- Data Maintenance: Web sources can change format, which might break scrapers. We should structure our code to handle minor changes and log failures clearly so we can fix promptly. Using community-maintained libraries (pybaseball, and so on) offloads some of this maintenance risk.
- Accuracy of Projections: MLB is a high-variance sport; even the best projections have a lot of uncertainty for a single game. Our model will provide a mean expectation, but actual outcomes will vary widely. We should communicate that these projections are averages, and consider incorporating a measure of volatility (for example, standard deviation of past scores) for use in simulation.
- Accuracy of Ownership: As discussed, our ownership estimates will only be approximate. Given how complex it is to model (reddit.com), we might treat our ownership output as a rough guide. Over time, if we gather actual ownership data, we can improve this component with more rigorous modeling or calibration.
- Integration with Optimizer: We must ensure the final output exactly matches what the optimizer or simulator expects. This might involve trial and error (for example, verifying that the optimizer correctly picks up our projection column). We will tailor our output accordingly.
- Timeline: Building this end-to-end will take substantial initial effort (data scraping scripts, model training, and so on). However, once set up, daily operation can be largely automated. The user should be prepared for an ongoing commitment to maintain the data and model.

In conclusion, the plan is to build a custom MLB DFS projection system with data scraped from FanGraphs and RotoWire, using a linear regression-based model for fantasy point projections and a heuristic model for ownership. The outputs will integrate into the current DFS optimizer and simulation workflow by augmenting the standard DraftKings player CSV with our projections and ownership. All components of this approach are achievable with available technology and data sources. With careful implementation and testing, this pipeline can provide tailored projections and ownership predictions for each MLB DFS slate, fulfilling the user's requirements.

## Sources

- FanGraphs data access: FanGraphs lacks a public API, but its static pages can be scraped with Python (angelineprotacio.com). Tools like PyBaseball already scrape FanGraphs so you do not have to (github.com). Game logs by player and year are available for detailed stats (billpetti.github.io).
- RotoWire lineups: RotoWire's daily MLB lineup page is a dependable source for starting lineups, which can be parsed programmatically (mattgorb.github.io). This allows integrating confirmed lineup and batting order data into projections.
- DraftKings CSV: DraftKings provides a downloadable player list CSV for each slate (including names and IDs) (reddit.com), which we will use as the template to insert our projections.
- DFS projection feasibility: Community discussions note that building custom projections is doable with enough data and effort, even if many use publicly available baselines (reddit.com). We leverage three seasons of stats for a solid foundation.
- Ownership modeling: DFS experts acknowledge ownership is complex and often adjusted by many situational factors (reddit.com). Our approach will incorporate key factors and possibly simulate lineup builds to approximate ownership levels.
