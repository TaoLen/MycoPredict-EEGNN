import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(APP.parent))


def main():
    source = APP.read_text(encoding="utf-8")
    assert "PCA" not in source
    assert "UMAP" not in source
    assert "density_column = st.container()" in source
    assert "structure_column, neighbor_table_column = st.columns(" in source
    assert "white-space: nowrap;" in source
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    app.slider[0].set_value(2)
    app.button[0].click()
    app.run(timeout=120)
    assert not app.exception, app.exception
    assert len(app.get("arrow_vega_lite_chart")) == 1
    domain_tables = [
        element.value
        for element in app.dataframe
        if "KDE domain status" in element.value.columns
    ]
    assert len(domain_tables) == 1
    assert list(domain_tables[0].columns) == [
        "Target",
        "KDE domain status",
        "Mean 5-NN embedding distance",
        "KDE CDF percentile",
        "Gaussian KDE density",
        "Labeled embedding references",
    ]
    forbidden_columns = {"AD status", "Structural zone", "Embedding zone"}
    assert forbidden_columns.isdisjoint(domain_tables[0].columns)

    assert not any(
        widget.label == "Embedding visualization" for widget in app.selectbox
    )

    applicability_selectbox = next(
        widget
        for widget in app.selectbox
        if widget.label == "Endpoint for domain details"
    )
    for task_index in range(9):
        applicability_selectbox.set_value(task_index)
        app.run(timeout=120)
        assert not app.exception, app.exception
        assert len(app.get("arrow_vega_lite_chart")) == 1
        applicability_selectbox = next(
            widget
            for widget in app.selectbox
            if widget.label == "Endpoint for domain details"
        )
    print("Streamlit Gaussian-KDE domain test passed for all endpoints.")


if __name__ == "__main__":
    main()
