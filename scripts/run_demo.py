from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.pipeline import VideoMemoPipeline
from videomemo.config import VideoMemoConfig

if __name__ == '__main__':
    cfg = VideoMemoConfig.load('configs/default.yaml')
    pipe = VideoMemoPipeline(cfg)
    print(pipe.run_and_export('data/raw/sample.mp4'))
