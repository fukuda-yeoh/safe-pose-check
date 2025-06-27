# 環境の作成
1.　プロジェクトファイルのターミナルでuvのインストール

```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

2.　仮想環境の作成

```uv sync```

3.　scoopのインストール

```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```

```Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression```

4.　ffmpegのインストール

```scoop install ffmpeg```

# システムの実行
```uv run main_live.py```
