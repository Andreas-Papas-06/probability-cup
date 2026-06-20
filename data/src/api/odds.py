def odds_api(API_KEY):
    import requests
    from collections import defaultdict



    url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"

    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    sports = requests.get(url, params=params).json()

    def parse_odds(events, aggregate="average"):
        """
        Parse The Odds API h2h output into:
            { date_str: { (home, away): {home: odds, away: odds, "Draw": odds} } }

        date_str : UTC date 'YYYY-MM-DD' taken from commence_time
        aggregate: how to collapse multiple bookmakers into one price per outcome
                "average" -> mean across books (rounded to 3 dp)
                "max"     -> best (highest) decimal price
                "first"   -> first bookmaker listed
        """
        NAME_FIXES = {
            "Cura\u00e7ao": "Curacao",
        }

        def fix_name(name):
            return NAME_FIXES.get(name, name)

        result = defaultdict(dict)

        for event in events:
            date_str = event["commence_time"][:10]       # 'YYYY-MM-DD'
            home = fix_name(event["home_team"])
            away = fix_name(event["away_team"])
            key = (home, away)

            # gather every price per outcome name across all bookmakers
            prices = defaultdict(list)
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market["outcomes"]:
                        prices[fix_name(outcome["name"])].append(outcome["price"])

            if not prices:
                continue

            odds = {}
            for name, vals in prices.items():
                if aggregate == "average":
                    odds[name] = round(sum(vals) / len(vals), 3)
                elif aggregate == "max":
                    odds[name] = max(vals)
                else:  # "first"
                    odds[name] = vals[0]

            result[date_str][key] = odds

        return dict(result)


    # usage
    parsed = parse_odds(sports)    

    for game in parsed['2026-06-21']:
        print(game)

