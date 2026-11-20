FFN 推論ランチャー

ファイルの配置と実行環境

main_inference.py と FFNInference3.py を同じディレクトリに配置してください。
repo_urakubo/ がある FFN の作業ディレクトリからコマンドを実行します。
JSON 内の相対パスは、コマンドを実行した作業ディレクトリを基準に解決されます。
すでに FFN が動作している Python の仮想環境を使用してください。

画像の準備と書き出しには numpy、h5py、opencv-python、tifffile、Pillow が必要です。
FFN 推論には repo_urakubo/run_inference.py の実行環境を使用します。
fastmorph は fill_hole=true を指定する場合に必要です。

最初に inference_config.json を編集してください。
target_image_dir、target_image_file_h5、train_dir、model_args、
ffn_inference_folder、output_segmentation_folder を、使用する画像、モデル、保存先に合わせます。
設定例のモデルと推論しきい値は、ご提供のランチャーの Sparse Z 設定を基にしています。

各段階を個別に実行する

  python main_inference.py --config inference_config.json --stage prepare
  python main_inference.py --config inference_config.json --stage inference
  python main_inference.py --config inference_config.json --stage postprocess

prepare は画像ファイルから HDF5 を作成します。
inference は作成済みの HDF5 を使って FFN 推論を実行します。
postprocess は推論結果を読み込み、画像ファイルへ書き出します。

3 段階を連続して実行する

  python main_inference.py --config inference_config.json --stage all

既存の HDF5 を置き換える

  python main_inference.py --config inference_config.json --stage prepare --overwrite-hdf5

設定した領域の既存結果を削除して推論し直す

  python main_inference.py --config inference_config.json --stage inference --clear-existing

既存の HDF5 を使って推論コマンドを確認する

  python main_inference.py --config inference_config.json --stage inference --clear-existing --dry-run

既存の推論結果を使って書き出し設定を確認する

  python main_inference.py --config inference_config.json --stage postprocess --dry-run

--dry-run は推論や画像の書き出しを実行せず、設定と対象を表示します。

HDF5 の作成と座標

入力は、2 次元のスライス画像を並べたディレクトリです。
TIFF スライスはグレースケール画像を使用してください。
その他の対応形式では、RGB/RGBA 画像をグレースケールに変換します。
ファイル名内の数字を数値として並べるため、z2 は z10 より前になります。

HDF5 には画像を ZYX 順で保存し、元の画素のデータ型を使用します。
平均強度、標準偏差、Start/End X/Y/Z、配列の軸順序、入力ファイル名も保存します。
統計値は保存する画像全体の全ボクセルから float64 精度で計算します。
標準偏差は母標準偏差です。

画像スタック全体を HDF5 に保存します。bounds は、その中で推論する領域を指定します。
Start 座標を含み、End 座標は含みません。
例えば Start X=10、End X=30 は、X 方向に 20 ボクセルの領域を指定します。
bounds の各値が null の場合、Start は 0、End は画像の各軸のサイズになります。
画像ファイル名に含まれる番号から、外部のグローバル座標原点を設定する処理はありません。

正規化と座標の設定優先順位

JSON に null 以外の値がある場合、その値を優先します。
null の場合は、HDF5 に保存されたメタデータを使用します。
古い HDF5 で座標が保存されていない場合は、画像サイズを使用します。
統計値が保存されていない場合は、HDF5 内の画像から計算します。
従来のスカラーデータセット名 mean、std も読み込めます。

options.image_mean と options.image_stddev は、学習用 JSON と共通のキーです。
推論用の設定例は null なので、入力画像から求めた値を使用します。
学習時の正規化を使用する場合は、その平均強度と標準偏差を数値で入力してください。

prepare 時に手動値を指定すると、採用値と画像から計算した値の両方を HDF5 に保存します。
HDF5 作成後に JSON を変更すると、その値を次回の推論で使用し、inference_run.json に記録します。
HDF5 のメタデータを更新する場合は、prepare を再実行してください。

画像から求めた標準偏差が 0 の場合も、その値を保存します。
推論には正の標準偏差が必要なので、一定強度の画像では正の手動値を指定してください。

チェックポイントの選択

options.train_dir に学習済みチェックポイントのディレクトリを指定します。
inference.ckpt=null の場合、model.ckpt-N の N が最大のものから順に調べ、
空でない .index ファイルと、必要な .data シャードがすべて揃ったものを採用します。
最大番号のファイル一式が未完成の場合は、その候補を飛ばします。
FFN がモデルのグラフを構築するため、.meta ファイルは任意です。

例えば ckpt に 38420160 を指定すると、train_dir/model.ckpt-38420160 を選択します。
チェックポイントのパスを文字列で直接指定することもできます。
相対パスの場合は、実行時の作業ディレクトリが基準です。
.index や .data の拡張子を付けず、チェックポイントのプレフィックスを指定してください。
ここではファイル構成を確認し、内容の妥当性は復元時に TensorFlow が検査します。

既存の推論結果

inference.clear_existing=true、または --clear-existing を指定すると、
選択した Start X/Y/Z に対応する結果を推論直前に削除します。
対象は .npz、.prob、.cpoint で、現在の x/y/seg-x_y_z 形式と従来の直下配置の両方に対応します。
削除対象は選択した領域のファイルです。他の領域や学習用チェックポイントは削除対象に含まれません。

削除を指定せずに対応する結果が存在する場合、説明を表示して停止します。
既存結果の画像書き出しには、postprocess を単独で実行できます。
推論パラメータと使用したチェックポイント、座標は、出力ディレクトリの
inference_params.pbtxt と inference_run.json に記録します。

後処理と画像の書き出し

FFN の標準的な結果は圧縮 NPZ で、segmentation キーにラベル配列が保存されています。
その配列を読み込みます。オブジェクト型の origins を読むために pickle を有効にする必要はありません。
単独の NPY ファイルを読み込む場合は、postprocess.result_file にパスを指定してください。
result_file=null の場合は、設定した領域に対応する NPZ を探します。

入力ラベル配列の順序は ZYX、書き出す各スライスの順序は YX です。
fill_hole=true の場合は、ご提供のランチャーと同じ設定で fastmorph の穴埋めを実行します。
設定例では false です。有効にするとラベル配列に後処理を加えます。
ラベルボリュームの作業用コピーを保持できるメモリが必要です。

32 bit グレースケール TIFF（既定値）
  output_filetype="tif", coloring=false, label_dtype="uint32"

16 bit グレースケール TIFF
  output_filetype="tif", coloring=false, label_dtype="uint16"

16 bit グレースケール PNG
  output_filetype="png", coloring=false, label_dtype="uint16"

カラー TIFF または PNG
  output_filetype="tif" または "png", coloring=true

カラー画像は uint8 の RGB 画像で、配列の順序は YXC です。
ラベル ID に基づく関数で色を決めるため、同じ ID は各スライスで同じ色になります。
背景のラベル 0 は黒です。すべての ID に異なる色が割り当てられる保証はありません。
元のラベル ID は NPZ/NPY に保持されます。

グレースケール画像では整数のラベル ID を保持し、変換前に最大値を検査します。
uint16 では 65535、uint32 では 4294967295 を超える ID があると停止します。
このランチャーの 32 bit グレースケール出力には TIFF を使用してください。

スライスのファイル名は z000000.tif または z000000.png の形式です。
HDF5 から領域を決めた場合は、その領域の Start Z を番号の開始値にします。
NPY/NPZ のパスを明示した場合は、0 から番号を付けます。
出力先に同名の画像ファイルがある場合は上書きします。

検証範囲

各軸のサイズが異なる人工画像とラベル配列で検証しました。
スライスの並び順、HDF5 のデータ型と統計値、メタデータ、手動設定の優先、
最大番号の完成済みチェックポイントの選択、protobuf リクエストの解析、
対象領域だけの結果削除、16 bit PNG、32 bit TIFF、RGB TIFF/PNG、
ラベル ID の上限超過時の停止を確認しています。
コマンドラインからの各段階の実行も、模擬推論プログラムで確認しました。
この Windows 作業環境では、実際の GPU FFN 推論と fastmorph の穴埋めは未実行です。

参照した FFN ソース

https://github.com/google/ffn/blob/master/run_inference.py
https://github.com/google/ffn/blob/master/ffn/inference/storage.py
https://github.com/google/ffn/blob/master/ffn/inference/inference.proto
