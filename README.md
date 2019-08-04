# なにこれ?
- Python(Flask)で稼働する顔画像解析Webアプリ
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
- [Generator]()
- [Encoder]()
- [FaceAnalyzer](https://www.dropbox.com/s/dhw4xe0txj1he6k/cnn.pt?dl=0)
ダウンロードしたモデルは./modelsの中に格納してください
---
