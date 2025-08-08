from pathlib import Path
from bs4 import BeautifulSoup
import sys
import types

pandas_stub = types.SimpleNamespace(DataFrame=object)
sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda *args, **kwargs: []))
sys.modules.setdefault("joblib", types.SimpleNamespace(Parallel=lambda *args, **kwargs: None, delayed=lambda f: f))
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        AutoTokenizer=object,
        AutoModelForSequenceClassification=object,
        AutoConfig=object,
        pipeline=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault("numpy", types.SimpleNamespace())
sklearn_stub = types.ModuleType("sklearn")
sklearn_metrics_stub = types.ModuleType("sklearn.metrics")
sklearn_model_selection_stub = types.ModuleType("sklearn.model_selection")
sklearn_model_selection_stub.train_test_split = lambda *args, **kwargs: ([], [])
sklearn_linear_model_stub = types.ModuleType("sklearn.linear_model")
sklearn_linear_model_stub.LogisticRegression = object
sklearn_feature_extraction_stub = types.ModuleType("sklearn.feature_extraction")
sklearn_feature_extraction_text_stub = types.ModuleType("sklearn.feature_extraction.text")
sklearn_feature_extraction_text_stub.TfidfVectorizer = object
sklearn_feature_extraction_stub.text = sklearn_feature_extraction_text_stub
sklearn_stub.metrics = sklearn_metrics_stub
sklearn_stub.model_selection = sklearn_model_selection_stub
sklearn_stub.linear_model = sklearn_linear_model_stub
sklearn_stub.feature_extraction = sklearn_feature_extraction_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.metrics", sklearn_metrics_stub)
sys.modules.setdefault("sklearn.model_selection", sklearn_model_selection_stub)
sys.modules.setdefault("sklearn.linear_model", sklearn_linear_model_stub)
sys.modules.setdefault("sklearn.feature_extraction", sklearn_feature_extraction_stub)
sys.modules.setdefault("sklearn.feature_extraction.text", sklearn_feature_extraction_text_stub)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import pipeline


def test_run_steps_adds_tag(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    xml_file = corpus / "sample.xml"
    xml_file.write_text("<RECORD></RECORD>")

    # ensure logs are written inside tmp_path
    monkeypatch.setattr(pipeline, "LOGS_PATH", tmp_path / "logs")

    def add_tag(soup: BeautifulSoup):
        pipeline.tdm_parser.modify_tag(soup, "test_tag", "1")
        return soup

    stats, failures = pipeline.run_steps(
        corpus_dir=corpus,
        steps=[(add_tag, {})],
        log_file_name="test_run",
    )

    assert stats["processed"] == 1
    assert failures == []
    assert "<test_tag>1</test_tag>" in xml_file.read_text()
