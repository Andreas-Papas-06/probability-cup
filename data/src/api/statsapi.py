import requests
import time
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
import pandas as pd


class Statsapi:
    def __init__(self, api_key):
        self.competions = {'comp_9389':'ASEAN Championship',
                        'comp_1554':'Africa Cup of Nations',
                        'comp_83579':'Africa Cup of Nations Qual.',
                        'comp_013219':'Arab Cup',
                        'comp_1376':'CONCACAF Gold Cup',
                        'comp_193547':'CONCACAF Nations League',
                        'comp_5749':'Copa América',
                        'comp_2949':'EURO',
                        'comp_3759':'EURO, Qualification',
                        'comp_6107':'FIFA World Cup',
                        'comp_9346':'Gulf Cup',
                        'comp_29967':'International Friendly Games',
                        'comp_574977':'UEFA Nations League',
                        'comp_8973':'World Championship Qual. AFC',
                        'comp_5720':'World Championship Qual. CAF',
                        'comp_0836':'World Championship Qual. CONCACAF', 
                        'comp_4682':'World Championship Qual. CONMEBOL',
                        'comp_7363':'World Championship Qual. OFC',
                        'comp_2954':'World Championship Qual. UEFA'}
        
        self.target = [ 'match_id', 'comp_id', 'season_id', 'date', 'home_name', 'home_id',
                        'away_name', 'away_id', 'home_score', 'away_score', 'home_xg_ft', 'away_xg_ft',
                        'home_sot_ft', 'away_sot_ft', 'home_sot_1h', 'away_sot_1h', 'home_sot_2h', 'away_sot_2h',
                        'home_shots_ft', 'away_shots_ft', 'home_shots_1h', 'away_shots_1h', 'home_shots_2h', 'away_shots_2h',
                        'home_corners_ft', 'away_corners_ft', 'home_corners_1h', 'away_corners_1h', 'home_corners_2h', 'away_corners_2h',
                        'home_fouls_ft', 'away_fouls_ft', 'home_offsides_ft', 'away_offsides_ft',
                        'home_offsides_1h', 'away_offsides_1h', 'home_offsides_2h', 'away_offsides_2h',
                        'home_yellows_ft', 'away_yellows_ft', 'home_yellows_1h', 'away_yellows_1h', 'home_yellows_2h', 'away_yellows_2h',
                        'home_poss_ft', 'away_poss_ft', 'home_poss_1h', 'away_poss_1h', 'home_poss_2h', 'away_poss_2h',
                        'home_reds_ft', 'away_reds_ft', 'home_reds_1h', 'away_reds_1h', 'home_reds_2h', 'away_reds_2h',
                        'ref_id', 'ref_name', 'home_score_1h', 'away_score_1h',]
    
        self.headers = {"Authorization": api_key}

    def get_match_wc(self, start_date='2026-06-01', end_date='2026-09-01', comp='comp_6107'):
        matches = []

        params = {"per_page": 100, "competition_id": comp, "date_from": start_date, "date_to": end_date}
        try:
            response = requests.get(f"https://api.thestatsapi.com/api/football/matches", headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"{self.competions[comp]}: {e}")
            return
        matches.extend(data.get("data", []))
        return matches

    
    def get_matches(self):
        matches = []
        
        for comp in self.competions:
            page = 1

            while True:
                params = {"page": page, "per_page": 100, "competition_id": comp, "date_from": "2021-01-01"}
                try:
                    response = requests.get(f"https://api.thestatsapi.com/api/football/matches", headers=self.headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    print(f"{self.competions[comp]}: {e}")
                    break
                matches.extend(data.get("data", []))
                pages = data.get("meta", {}).get("total_pages", page)
                time.sleep(5.1)

                if page >= pages: 
                    break
                else:
                    page += 1

        return matches
    
    def get_match_stats(self, matches):

        def seg(stats, key, segment):
            """Pull (home, away) for one stat + segment. Returns (None, None) if absent."""
            block = stats.get(key)
            if not block:
                return None, None
            s = block.get(segment)          # 'all' / 'first_half' / 'second_half'
            if not s:                       # null half, or missing
                return None, None
            return s.get("home"), s.get("away")

        rows = []
        count = 0

        for match in matches:
            match_id = match['id']
            comp_id = match['competition_id']
            season_id = match['season_id']
            date = match['utc_date']
            home_name = match['home_team']['name']
            home_id = match['home_team']['id']
            away_name = match['away_team']['name']
            away_id = match['away_team']['id']
            home_score = match['score']['home']
            away_score = match['score']['away']

            row = {'match_id':match_id, 
                  'comp_id':comp_id, 
                  'season_id':season_id,
                  'date':date,
                  'home_name':home_name, 
                  'home_id':home_id, 
                  'away_name':away_name, 
                  'away_id':away_id,
                  'home_score':home_score,
                  'away_score':away_score}
            
            try:
                response = requests.get(f"https://api.thestatsapi.com/api/football/matches/{match_id}/stats", headers=self.headers)
                response.raise_for_status()
                data = response.json().get("data", {})
            except Exception as e:
                count += 1
                data = {} 

            stats = {}
            for section in ("overview", "shots", "attack", "passes",
                            "defending", "goalkeeping"):
                stats.update(data.get(section, {}))

            wanted = {
                    "xg":       "expected_goals",
                    "sot":      "shots_on_target",
                    "shots":    "total_shots",
                    "corners":  "corner_kicks",
                    "fouls":    "fouls",
                    "offsides": "offsides",
                    "yellows":  "yellow_cards",
                    "poss":     "ball_possession",
                    "reds":     "red_cards"
                    }
            
            for out, key in wanted.items():
                for segment, tag in (("all", "ft"), ("first_half", "1h"), ("second_half", "2h")):
                    h, a = seg(stats, key, segment)
                    row[f"home_{out}_{tag}"] = h
                    row[f"away_{out}_{tag}"] = a
            rows.append(row)
            time.sleep(5.1)

        print(count)
        return rows
    
    def get_match_deatails(self, matches):
        rows = []

        for match in matches:
            match_id = match['id']

            try:
                response = requests.get(f"https://api.thestatsapi.com/api/football/matches/{match_id}", headers=self.headers)
                response.raise_for_status()
                data = response.json().get("data", {})
            except Exception as e:
                #print(f"{match_id}: {e}")
                data = {} 

            ref = data.get("referee") or {}            # None-safe
            score = data.get("score") or {}
            row = {
                "match_id":      match_id,
                "ref_id":        ref.get("id"),
                "ref_name":      ref.get("name"),       # grab the name too — useful
                "home_score_1h": score.get("half_time_home"),
                "away_score_1h": score.get("half_time_away"),
            }
            rows.append(row)
            time.sleep(5.1)

        return rows
    
    def get_games_today(self):
        matches = self.get_match_wc()

        eastern = ZoneInfo("America/New_York")
        today = datetime.now(eastern).date()

        start = datetime(today.year, today.month, today.day, 5, tzinfo=eastern)
        end = start + timedelta(days=1)
        start_utc = start.astimezone(timezone.utc)
        end_utc   = end.astimezone(timezone.utc) 
        
        matches = [match for match in matches if start_utc <= datetime.fromisoformat(match['utc_date']) <= end_utc]
        details = self.get_match_deatails(matches)
        for m, d in zip(matches, details):
            m['ref_id'] = d['ref_id']

        return matches

    def update_last_day_results(self, date=None, games=[]):
        matches = self.get_match_wc()
        if games:
            matches = [match for match in matches if match['id'] in games]
        else:
            today = pd.to_datetime(date, format="%Y%m%d")
            yesterday_dt = pd.to_datetime(date, format="%Y%m%d") - timedelta(days=1)
            yesterday = yesterday_dt.strftime("%Y%m%d") 

            eastern = ZoneInfo("America/New_York")
            today = pd.to_datetime(date, format="%Y%m%d")

            start = datetime(today.year, today.month, today.day, 5, tzinfo=eastern) - timedelta(days=1)
            end = start + timedelta(days=1)
            start_utc = start.astimezone(timezone.utc)
            end_utc   = end.astimezone(timezone.utc) 
            
            matches = [match for match in matches if start_utc <= datetime.fromisoformat(match['utc_date']) <= end_utc]

        stats = pd.DataFrame(self.get_match_stats(matches))
        details = pd.DataFrame(self.get_match_deatails(matches))
        combined = pd.merge(stats, details, on="match_id", how="left")
        df = combined[self.target]
        if games:
            df_old = pd.read_csv(f'../data/data_logs/data_sets_clean/dixon_coles_{date}_inputs.csv', )
            df_new = pd.concat([df_old, df], ignore_index=True)
            df_new[self.target].to_csv(f'../data/data_logs/data_sets_clean/dixon_coles_{date}_inputs.csv', index = False)
        else:
            df_old = pd.read_csv(f'../data/data_logs/data_sets_clean/dixon_coles_{yesterday}_inputs.csv', )
            df_new = pd.concat([df_old, df], ignore_index=True)
            df_new[self.target].to_csv(f'../data/data_logs/data_sets_clean/dixon_coles_{date}_inputs.csv', index = False)
        return df
        

