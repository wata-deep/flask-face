# なにこれ?
- Python(Flask)で稼働する顔画像解析Webアプリ
    - バックエンドはPyTorch
    - 顔画像変換も実装予定
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
    - [Generator](https://www.dropbox.com/s/sum9a25xqn8ubsu/gen.pt?dl=0)
    - [Encoder](https://www.dropbox.com/s/v1zs7nnkinjh4ik/enc.pt?dl=0)
    - [FaceAnalyzer](https://www.dropbox.com/s/dhw4xe0txj1he6k/cnn.pt?dl=0)
---
