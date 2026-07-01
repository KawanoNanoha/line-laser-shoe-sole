# 中央値から外れた点を除外
        # y_median = np.median(y)
        # y_std = np.std(y)
        # mask_filter = np.abs(y - y_median) < y_std * 1.5
        # x = x[mask_filter]
        # y = y[mask_filter]

        # # x範囲が広すぎる断面はスキップ（横線混入）
        # if x.max() - x.min() > 250:
        #     continue