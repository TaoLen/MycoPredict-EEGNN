import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(APP.parent))


def main():
    started = time.perf_counter()
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    elapsed = time.perf_counter() - started
    assert not app.exception, app.exception
    assert "single_prediction" not in app.session_state
    assert len(app.get("arrow_vega_lite_chart")) == 0
    assert not any(
        element.label == "Embedding status" for element in app.metric
    )
    print(f"Streamlit initial render passed in {elapsed:.3f} seconds.")


if __name__ == "__main__":
    main()
