# sandbox-openapi-whisper

- `pyproject.toml` を開いて環境・バージョンにあった pytorch の url に差し替える.
- `src/main.py` 内を直に編集して指定したいaudioファイルへのパスに変更する.

```sh
poetry install
cd src
poetry run main.py
```

- `<audio-file-name>.txt`, `<audio-file-name>.vtt` 内を直に編集して指定したいaudioファイルへのパスに変更する.
