# y行ごとにxの平均を取得
    # points = []
    # for y in range(filtered_mask.shape[0]):
    #     xs = np.where(filtered_mask[y, :] > 0)[0]
    #     if len(xs) > 0:
    #         points.append((int(np.mean(xs)), y))