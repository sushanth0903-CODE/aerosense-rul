import pandas as pd

from src.data_pipeline import add_train_rul, get_informative_features


def test_train_rul_is_computed_and_clipped():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )
    enriched = add_train_rul(df, clip_max=2)
    assert enriched.loc[0, "RUL"] == 2
    assert enriched.loc[2, "RUL"] == 0
    assert enriched.loc[3, "RUL"] == 1
