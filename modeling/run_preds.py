import pandas as pd
from models.predict import prepare_df, fit_all_models, red_rates, predict_fixture
from models.referee import build_referee_table, adjust_cards_fouls, penalty_prediction

def run_preds(path, matches):
    preds = []
    raw = pd.read_csv(path, index_col=0)        
    df = prepare_df(raw)

    models, shares = fit_all_models(df)        
    rrates, rbase  = red_rates(df)              
    ref_table      = build_referee_table(df)     

    for match in matches:
        home_id = match['home_team']['id']
        away_id = match['away_team']['id']
        this_match_ref_id = match['ref_id']

        pred = predict_fixture(models, shares, home_id, away_id, neutral=True,
                            red_rates_=rrates, red_base=rbase)

        # 4. APPLY referee adjustments  (scales cards/fouls, real penalty)
        pred = adjust_cards_fouls(pred, ref_table, this_match_ref_id)
        pred["penalty"] = penalty_prediction(ref_table, this_match_ref_id)
        preds.append(pred)
    return preds, models, shares, rrates, rbase, ref_table

    

    