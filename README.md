# ryoden-numazu-hvac-optimize

# 環境構築

(`Python 3.11.9` で動作確認済み)

## 1. インストールとセットアップ

1. リポジトリをクローンします：

   ```bash
   git clone https://github.com/yukiatsunori/Actine_simlation.git
   cd Actine_simulation
   ```

2. `uv` をインストール(uv がない場合):

   ```
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. `uv` で仮想環境を作成(Python 3.11 を使用):

   ```
   uv venv --python 3.11
   ```

   `uv` で依存パッケージをインストール:

   ```
   uv sync
   ```


### シミュレーションを実行:

```bash
uv run main.py 
```

### 可視化:

```bash
uv run visualization/visualize_module.py
```