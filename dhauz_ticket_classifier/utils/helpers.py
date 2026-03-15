"""Helpers reutilizáveis para scripts: carregamento de LLM remoto/local, inferência de classes e salvamento de reports."""
from pathlib import Path
from typing import List, Optional

def make_remote_llm(remote_url: str, remote_key: Optional[str] = None):
    try:
        import requests
    except Exception:
        requests = None

    class RemoteLLM:
        def __init__(self, url: str, api_key: Optional[str] = None):
            self.url = url
            self.api_key = api_key
            self._requests = requests

        def generate(self, prompts, batch_size=1):
            if self._requests is None:
                raise RuntimeError('requests not available; install requests')
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            results = []
            for prompt in prompts:
                payload = {'inputs': prompt}
                resp = self._requests.post(self.url, headers=headers, json=payload, timeout=60)
                try:
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    data = {'error': str(e)}
                if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                    results.append(data)
                elif isinstance(data, dict) and 'generated_text' in data:
                    results.append({'generated_text': data['generated_text']})
                else:
                    results.append({'generated_text': str(data)})
            return results

    if not remote_url:
        raise ValueError('remote_url is required')
    return RemoteLLM(remote_url, api_key=remote_key)


def maybe_load_llm(use_llm: bool, use_remote: bool, remote_url: Optional[str] = None, remote_key: Optional[str] = None, model_name: Optional[str] = None):
    if use_remote:
        return make_remote_llm(remote_url, remote_key)
    if use_llm:
        if model_name is None:
            raise ValueError('Local LLM requested but no model_name provided')
        from ..models.llm import LLMPipeline
        return LLMPipeline(model_name=model_name)
    return None


def infer_classes(data_csv: str = 'data/dataset_processed.csv', vector_store=None) -> List[str]:
    classes = None
    p = Path(data_csv)
    if p.exists():
        try:
            import pandas as pd
            df = pd.read_csv(p)
            if 'class' in df.columns:
                classes = sorted(df['class'].unique())
        except Exception:
            classes = None

    if not classes and vector_store is not None:
        try:
            cols = set()
            if getattr(vector_store, 'db', None) is not None:
                docs = vector_store.db.get(include=['metadatas'])
                for m in docs.get('metadatas', []):
                    if isinstance(m, dict) and 'class' in m:
                        cols.add(m['class'])
            if cols:
                classes = sorted(list(cols))
        except Exception:
            classes = None

    if not classes:
        classes = ['Miscellaneous']
    return classes


def save_classification_report(report_dict: dict, classes: List[str], out_dir: str = './results', prefix: str = 'classification_report') -> dict:
    import json
    from pathlib import Path
    out = {}
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    report_path = p / f"{prefix}.json"
    with open(report_path, 'w', encoding='utf8') as fh:
        json.dump(report_dict, fh, ensure_ascii=False, indent=2)
    class_map = {str(i): c for i, c in enumerate(classes)}
    mapping_path = p / f"{prefix}_class_map.json"
    with open(mapping_path, 'w', encoding='utf8') as fh:
        json.dump(class_map, fh, ensure_ascii=False, indent=2)
    out['report'] = str(report_path)
    out['class_map'] = str(mapping_path)
    return out
