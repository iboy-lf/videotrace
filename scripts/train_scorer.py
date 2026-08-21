from __future__ import annotations

from pathlib import Path
import json
import shutil
from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.scorer import train_scorer


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog='videomemo-train-scorer')
    parser.add_argument('examples', nargs='+', help='one or more scorer.json files')
    parser.add_argument('--model', default='outputs_train/scorer_model.pkl')
    parser.add_argument('--metrics', default='outputs_train/scorer_metrics.json')
    parser.add_argument('--sync-to', default='outputs/scorer_model.pkl')
    args = parser.parse_args()

    result = train_scorer(args.examples, args.model)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.dump(), ensure_ascii=False, indent=2), encoding='utf-8')

    sync_path = Path(args.sync_to)
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.model_path, sync_path)

    payload = result.dump()
    payload['synced_to'] = str(sync_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
