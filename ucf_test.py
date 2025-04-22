import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test(net, config, wind, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()  # モデルを評価モードに設定
        net.flag = "Test"
        
        # モデルファイルのロード（CPU用に指定）
        if model_file is not None:
            net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))  # GPUがない場合でもCPUでロード

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")  # Ground Truthをロード
        frame_predict = None
        
        cls_label = []  # 正解ラベル
        cls_pre = []  # 予測ラベル
        temp_predict = torch.zeros((0)).cpu()  # 予測結果を格納するテンソル
        
        # テストデータの数を確認
        num_test_data = len(test_loader.dataset)
        
        for i in range(num_test_data):
            _data, _label, _extra = next(load_iter)  # 3つ以上の値を返す場合

            _data = _data.cpu()  # データをCPUに移動
            _label = _label.cpu()  # ラベルをCPUに移動

            res = net(_data)  # モデルの予測結果
            a_predict = res["frame"]  # フレームごとの予測結果
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)  # 予測結果を結合
            
            # 10回ごとに評価
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))  # 正解ラベルを追加
                a_predict = temp_predict.mean(0).cpu().numpy()  # 平均予測結果を計算
                # print(a_predict)
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)  # 予測ラベルを0か1に変換

                # セグメントごとの平均予測値を出⼒
                segment_max_predict = np.max(a_predict) # セグメントごとの平均予測値
                if cls_pre[i // 10] == 1:
                    print(f"Segment{i // 10 + 1} predict: Abnormal score:{segment_max_predict*100:.10f}\n")
                else:
                    print(f"Segment{i // 10 + 1} predict: Normal score: {(1 -segment_max_predict)*100:.10f}\n")
                
                fpre_ = np.repeat(a_predict, 16)  # フレームごとの予測結果を複製
                if frame_predict is None:
                    frame_predict = fpre_  # 初回はそのまま代入
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  # それ以降は結合
                
                temp_predict = torch.zeros((0)).cpu()  # 次回の予測用にテンソルを初期化
                # print(frame_predict)
        
        # 使用したテストデータのフレーム数に合わせた評価
        fpr, tpr, _ = roc_curve(frame_gt[:len(frame_predict)], frame_predict)
        auc_score = auc(fpr, tpr)
        
        # 正解率の計算
        correct_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = correct_num / len(cls_pre)
        
        # 精度と再現率の計算
        precision, recall, _ = precision_recall_curve(frame_gt[:len(frame_predict)], frame_predict)
        ap_score = auc(recall, precision)

        # 可視化 (windオブジェクトの使用)
        if wind is not None:
            wind.plot_lines('roc_auc', auc_score)
            wind.plot_lines('accuracy', accuracy)
            wind.plot_lines('pr_auc', ap_score)
            wind.lines('scores', frame_predict)
            wind.lines('roc_curve', tpr, fpr)
        
        # テスト結果を保存
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)

        # 結果を表示
        print(f"ROC AUC Score: {auc_score}")
        print(f"Accuracy: {accuracy}")
        print(f"PR AUC Score: {ap_score}")


# テスト関数の実行部分（例）
if __name__ == "__main__":
    # 引数の解析
    args = parse_args()
    
    # 設定ファイルの読み込み
    config = Config(args)
    
    # モデルの初期化
    net = WSAD(input_size=config.len_feature, flag="Test", a_nums=60, n_nums=60)

    # CPU用に設定
    device = torch.device('cpu')  # GPUがなくてもCPUに設定
    net = net.cpu()  # モデルをCPUに移動

    # テストデータのローダー準備
    test_loader = data.DataLoader(
        UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal, num_segments=config.num_segments, len_feature=config.len_feature),
        batch_size=1, shuffle=False, num_workers=config.num_workers
    )

    # テスト情報を保持する辞書
    test_info = {
        "step": [],
        "auc": [],
        "ap": [],
        "ac": []
    }
    
    # モデルファイルのパスを指定してテスト関数を呼び出す
    model_file = os.path.join(args.model_path, "ucf_trans_2022.pkl")
    test(net, config, None, test_loader, test_info, step=0, model_file=model_file)