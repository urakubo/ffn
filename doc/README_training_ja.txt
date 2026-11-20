共通 JSON 版

FFN 作業ディレクトリで実行してください。相対パスは実行時の作業ディレクトリ基準です。
repo/configs/ffn_shared_config_4gpu.json（または 2gpu）を作業ディレクトリへ、
main_prep_train.py と main_distributed_train.py を repo/ へ置きます。

同じ JSON を両方のプログラムに指定します。
  options: 学習データ、モデル、正規化、保存先などの共通設定と PS/worker のアドレス。
  environment: 共通の環境変数。
  preparation: 入力画像の glob、ボリューム別の partition/TFRecord 出力、準備パラメータ。
  single_gpu: 単一 GPU の番号。
  processes: 分散学習のプロセス別設定。
各プログラムは使わないセクションを無視します。options の学習フラグは下流コードの名前に合わせます。
分散版は実行時に未登録の JSON フラグを無視します。dry-run は起動前の候補引数を表示します。
旧分散 JSON と直接指定の CLI も利用できます。

まず確認（ファイル生成やトレーニングは行いません）:
  python repo/main_prep_train.py --config ffn_shared_config_4gpu.json --stage all --dry-run
  python repo/main_distributed_train.py --config ffn_shared_config_4gpu.json --role worker0 --dry-run

使う段階を選択:
  --stage hdf5                 HDF5 生成のみ
  --stage compute_partitions   パーティション計算のみ
  --stage build_coordinates    TFRecord 作成のみ
  --stage prepare              上記 3 段階を選択した各ボリュームに実施
  --stage train                既存データから train.py で単一 GPU 学習
  --stage all                  準備後に単一 GPU 学習
準備対象を絞る場合は --volume 2 などを追加します。複数指定できます。
--volume は準備対象のみを絞ります。学習には常に options のデータと TFRecord glob が使われます。
HDF5 は既定で既存ファイルへの上書きを拒否します。置換する場合だけ --overwrite-hdf5 を指定してください。
既存のデータで学習を再開するときは --stage train を使い、保存先を同時に使う別ジョブは停止してください。

正規化は options.image_mean / image_stddev（または image_offset_scale_map）を共有します。
元の単一 GPU ランチャーによる自動計算は行いません。サンプルは以前の分散コマンドの 125 / 74 です。
2番目のボリュームはファイル名 v2、TFRecord 内のタグ 2 にしています。既存記録に合わせてください。

分散実行は同じ JSON を指定し、別々の端末で --role server、worker0、worker1、必要に応じ worker2、worker3 を起動します。
