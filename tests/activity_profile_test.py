import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
SMILES = "CCc1ccc(C=CC(=O)c2cccc(Br)c2)cc1"
sys.path.insert(0, str(APP.parent))


def main():
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.text_area[0].set_value(SMILES)
    app.button[0].click().run()
    assert not app.exception, app.exception

    activity = app.dataframe[0].value
    assert list(activity.columns) == [
        "Target",
        "Prediction",
        "Threshold-adjusted score",
        "Probability",
        "Threshold",
        "Epistemic uncertainty",
    ]
    assert "Embedding KDE applicability domain" not in activity.columns

    dilution = activity.loc[
        activity["Target"] == "M. tuberculosis (dilution)"
    ].iloc[0]
    maba = activity.loc[
        activity["Target"] == "M. tuberculosis (MABA)"
    ].iloc[0]
    print(activity.to_string(index=False))
    assert dilution["Probability"] < dilution["Threshold"]
    assert dilution["Threshold-adjusted score"] < 0.5
    assert dilution["Prediction"] == "Inactive"
    assert maba["Probability"] > maba["Threshold"]
    assert maba["Threshold-adjusted score"] > 0.5
    assert maba["Prediction"] == "Active"
    print("Rendered activity profile passed.")


if __name__ == "__main__":
    main()
