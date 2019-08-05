# なにこれ?
- Python(Flask)で稼働する顔画像解析Webアプリ
    - バックエンドはPyTorch
    - 顔画像変換は強制ホラー画像化
---
# 使い方
- CPU
```sh
python server.py
```
- GPU
```sh
python server.py --cuda "device名"
```
---
# 必要なパッケージ
- Flask
- PyTorch
- OpenCV
- etc...
---
# 学習済みモデル
- ダウンロードしたモデルは./modelsの中に格納してください
- models
    - [Generator](https://www.dropbox.com/s/ur90hd510k68cci/gen.pt?dl=0)
    - [Encoder](https://www.dropbox.com/s/gt8fvcdxmfiaeen/enc.pt?dl=0)
    - [FaceAnalyzer](https://www.dropbox.com/s/dhw4xe0txj1he6k/cnn.pt?dl=0)
---
