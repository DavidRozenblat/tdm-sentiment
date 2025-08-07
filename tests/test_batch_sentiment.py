import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

# Stub torch to avoid heavy dependency during tests
torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules.setdefault('torch', torch_stub)

transformers_stub = types.SimpleNamespace(
    AutoTokenizer=object,
    AutoModelForSequenceClassification=object,
    AutoConfig=object,
    pipeline=lambda *args, **kwargs: None,
)
sys.modules.setdefault('transformers', transformers_stub)

bs4_stub = types.SimpleNamespace(BeautifulSoup=object, NavigableString=object)
sys.modules.setdefault('bs4', bs4_stub)

pandas_stub = types.SimpleNamespace()
sys.modules.setdefault('pandas', pandas_stub)

tqdm_stub = types.SimpleNamespace(tqdm=lambda *args, **kwargs: [])
sys.modules.setdefault('tqdm', tqdm_stub)

joblib_stub = types.SimpleNamespace(Parallel=lambda *args, **kwargs: None, delayed=lambda f: f)
sys.modules.setdefault('joblib', joblib_stub)

numpy_stub = types.SimpleNamespace(array=lambda *args, **kwargs: None)
sys.modules.setdefault('numpy', numpy_stub)

sklearn_stub = types.ModuleType('sklearn')
sklearn_metrics_stub = types.ModuleType('sklearn.metrics')
sklearn_model_selection_stub = types.ModuleType('sklearn.model_selection')
sklearn_stub.metrics = sklearn_metrics_stub
sklearn_stub.model_selection = sklearn_model_selection_stub
sys.modules.setdefault('sklearn', sklearn_stub)
sys.modules.setdefault('sklearn.metrics', sklearn_metrics_stub)
sys.modules.setdefault('sklearn.model_selection', sklearn_model_selection_stub)

config_stub = types.ModuleType('config')
config_stub.tdm_parser_module = types.SimpleNamespace(TdmXmlParser=lambda: None)
config_stub.is_economic_module = types.SimpleNamespace(EconomicClassifier=object)
config_stub.tf_idf_extractor = types.SimpleNamespace(TfidfKeywordExtractor=object)
config_stub.sentiment_model = types.SimpleNamespace(TextAnalysis=object)
config_stub.logger = types.SimpleNamespace(Logger=object)
config_stub.FILE_NAMES_PATH = Path('.')
config_stub.LOGS_PATH = Path('.')
config_stub.TF_IDF_MODEL_PATH = Path('.')
config_stub.ROBERTA_MODEL_PATH = Path('.')
config_stub.BERT_MODEL_PATH = Path('.')
sys.modules.setdefault('config', config_stub)

from sentiment.sentiment_model.sentiment_model import TextAnalysis
from pipeline import article_average_sentiment_helper

class DummyPipeline:
    def __init__(self):
        self.calls = 0
    def __call__(self, inputs, return_all_scores=True):
        self.calls += 1
        templates = [
            [
                {'label': 'positive', 'score': 0.2},
                {'label': 'neutral', 'score': 0.1},
                {'label': 'negative', 'score': 0.7},
            ],
            [
                {'label': 'positive', 'score': 0.6},
                {'label': 'neutral', 'score': 0.2},
                {'label': 'negative', 'score': 0.2},
            ],
        ]
        return templates[:len(inputs)]

def make_analyzer():
    analyzer = TextAnalysis.__new__(TextAnalysis)
    analyzer.sentiment_pipeline = DummyPipeline()
    return analyzer

def test_batch_txt_sentiment_dict():
    analyzer = make_analyzer()
    texts = ['text one', 'text two']
    results = analyzer.batch_txt_sentiment_dict(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]['positive'] == 0.2
    assert analyzer.sentiment_pipeline.calls == 1

def test_article_average_sentiment_helper_uses_batch():
    analyzer = make_analyzer()
    paragraphs = ['para1', 'para2']
    avg = article_average_sentiment_helper(paragraphs, analyzer)
    assert avg == {'positive': 0.4, 'neutral': 0.15, 'negative': 0.45}
    assert analyzer.sentiment_pipeline.calls == 1