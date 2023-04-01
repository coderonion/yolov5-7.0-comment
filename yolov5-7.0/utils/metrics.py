# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""
# 导入Python 模块
import math # 数学函数
import warnings # 警告信息的控制
from pathlib import Path # 面向对象的文件系统路径
# 导入第三方库
import matplotlib.pyplot as plt # 图形展示库
import numpy as np # 数据处理库
import torch # 深度学习库     

# ----------------- 导入自定义的其他包 -------------------
from utils import TryExcept, threaded


def fitness(x):
    # Model fitness as a weighted combination of metrics
    """通过指标加权的形式返回适应度(最终mAP)  在train.py中使用
    Model fitness as a weighted combination of metrics
    判断模型好坏的指标不是mAP@0.5也不是mAP@0.5:0.95 而是[P, R, mAP@0.5, mAP@0.5:0.95]4者的加权
    一般w=[0,0,0.1,0.9]  即最终的mAP=0.1mAP@0.5 + 0.9mAP@0.5:0.95
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    # (torch.tensor).sum(1) 每一行求和tensor为二维时返回一个以每一行求和为结果(常数)的行向量
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10). [pred_sum, 10]=[1905, 10] bool 整个数据集所有图片中所有预测框在每一个iou条件下(0.5~0.95)10个是否是TP
        conf:  Objectness value from 0-1 (nparray).  [img_sum]=[1905] 整个数据集所有图片的所有预测框的conf 
        pred_cls:  Predicted object classes (nparray).  [img_sum]=[1905] 整个数据集所有图片的所有预测框的类别 这里的tp、conf、pred_cls是一一对应的
        target_cls:  True object classes (nparray). [gt_sum]=[929] 整个数据集所有图片的所有gt框的class
        plot:  Plot precision-recall curve at mAP@0.5 bool  
        save_dir:  Plot save directory  runs\train\exp
        names: dict{key(class_index):value(class_name)} 获取数据集所有类别的index和对应类名
    # Returns
        The average precision as computed in py-faster-rcnn.
         p[:, i]: [nc] 最大平均f1时每个类别的precision
         r[:, i]: [nc] 最大平均f1时每个类别的recall
         ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
         f1[:, i]: [nc] 最大平均f1时每个类别的f1
         unique_classes.astype('int32'): [nc] 返回数据集中所有的类别index
    """
    # 计算mAP 需要将tp按照conf降序排列
    # Sort by objectness
    i = np.argsort(-conf)  #按conf从大到小排序 返回数据对应的索引
    # 得到重新排序后对应的 tp, conf, pre_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True) # 对类别去重, 因为计算ap是对每类进行
    nc = unique_classes.shape[0]  # number of classes, number of detections 数据集类别数

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting  px: [0, 1] 中间间隔1000个点 x坐标(用于绘制P-Conf、R-Conf、F1-Conf) py: y坐标[] 用于绘制IOU=0.5时的PR曲线
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)) # 初始化 对每一个类别在每一个IOU阈值下 计算AP P R   ap=[nc, 10]  p=[nc, 1000] r=[nc, 1000]
    for ci, c in enumerate(unique_classes):   # ci: index 0   c: class 0  unique_classes: 所有gt中不重复的class
        i = pred_cls == c  #  # i: 记录着所有预测框是否是c类别框   是c类对应位置为True, 否则为False
        # n_l: gt框中的c类别框数量  = tp+fn   254
        n_l = nt[ci]  # number of labels
        # n_p: 预测框中c类别的框数量   695
        n_p = i.sum()  # number of predictions
        # 如果没有预测到 或者 ground truth没有标注 则略过类别c
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
         # Accumulate FPs(False Positive) and TPs(Ture Positive)   FP + TP = all_detections
            # tp[i] 可以根据i中的的True/False觉定是否删除这个数  所有tp中属于类c的预测框
            #       如: tp=[0,1,0,1] i=[True,False,False,True] b=tp[i]  => b=[0,1]
            # a.cumsum(0)  会按照对象进行累加操作
            # 一维按行累加如: a=[0,1,0,1]  b = a.cumsum(0) => b=[0,1,1,2]   而二维则按列累加
            # fpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下FP个数 最后一行表示c类在该iou阈值下所有FP数
            # tpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下TP个数 最后一行表示c类在该iou阈值下所有TP数
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall=TP/(TP+FN)  加一个1e-16的目的是防止分母为0
            # n_l=TP+FN=num_gt: c类的gt个数=预测是c类而且预测正确+预测不是c类但是预测错误
            # recall: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的召回率
        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的recall值  r=[nc, 1000]  每一行从小到大
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        # Precision=TP/(TP+FP)
        # precision: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的精确率
        precision = tpc / (tpc + fpc)  # precision curve
        # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的precision值  p=[nc, 1000]
        # 总体上是从小到大 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        # 对c类别, 分别计算每一个iou阈值(0.5~0.95 10个)下的mAP
        for j in range(tp.shape[1]): # tp [pred_sum, 10]
            # 这里执行10次计算ci这个类别在所有mAP阈值下的平均mAP  ap[nc, 10]
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5 用于绘制每一个类别IOU=0.5时的PR曲线

    # Compute F1 (harmonic mean of precision and recall)
    # 计算F1分数 P和R的调和平均值  综合评价指标
    # 我们希望的是P和R两个越大越好, 但是P和R常常是两个冲突的变量, 经常是P越大R越小, 或者R越大P越小 所以我们引入F1综合指标
    # 不同任务的重点不一样, 有些任务希望P越大越好, 有些任务希望R越大越好, 有些任务希望两者都大, 这时候就看F1这个综合指标了
    # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的f1值  f1=[nc, 1000]
    f1 = 2 * p * r / (p + r + eps)  # 用于绘制P-Confidence(F1_curve.png)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    # f1=[nc, 1000]   f1.mean(0)=[1000]求出所有类别在x轴每个conf点上的平均f1
    # .argmax(): 求出每个点平均f1中最大的f1对应conf点的index
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    用于ap_per_class函数中，计算某个类别在某个iou阈值下的mAP
    # Arguments
        recall:    The recall curve (list) (list) [1635] 在某个iou阈值下某个类别所有的预测框的recall  从小到大 (每个预测框的recall都是截至到这个预测框为止的总recall)
        precision: The precision curve (list)   (list) [1635] 在某个iou阈值下某个类别所有的预测框的precision 总体上是从大到小 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506  (每个预测框的precision都是截至到这个预测框为止的总precision)
                      
    # Returns
        Average precision, Average precision 返回某类别在某个iou下的mAP(均值) [1]
        precision curve,  precision curve [1637] 返回 开头 + 输入precision(排序后) + 末尾
        recall curve recall curve [1637] 返回 开头 + 输入recall + 末尾
    """

    # Append sentinel values to beginning and end   在开头和末尾添加保护值 防止全零的情况出现 value Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))  # [1637]
    mpre = np.concatenate(([1.0], precision, [0.0]))  # [1637]

    # Compute the precision envelope  np.flip翻转顺序  np.flip(mpre): 把一维数组每个元素的顺序进行翻转 第一个翻转成为最后一个
    # np.maximum.accumulate(np.flip(mpre)): 计算数组(或数组的特定轴)的累积最大值 令mpre是单调的 从小到大
    # 到这大概看明白了这步的目的: 要保证mpre是从大到小单调的(左右可以相同)
    # 我觉得这样可能是为了更好计算mAP 因为如果一直起起伏伏太难算了(x间隔很小就是一个矩形) 而且这样做误差也不会很大 两个之间的数都是间隔很小的
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp': # 用一些典型的间断点来计算AP
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)  [0, 0.01, ..., 1]
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate 计算两个list对应点与点之间四边形的面积 以定积分形式估算AP 第一个参数是y 第二个参数是x
    else:  # 'continuous'   # 采用连续的方法计算AP
         # 通过错位的方式 判断哪个点当前位置到下一个位置值发生改变 并通过！=判断 返回一个布尔数组
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        # 值改变了就求出当前矩阵的面积  值没变就说明当前矩阵和下一个矩阵的高相等所有可以合并计算
    return ap, mpre, mrec


class ConfusionMatrix:
    """用在val.py中计算混淆矩阵
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    混淆矩阵: 定义  更新  return  绘制  print打印
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        # 初始化混淆矩阵 pred x gt  其中横坐标/纵坐标第81类为背景类
        # 如果某个gt[j]没用任何pred正样本匹配到 那么[nc, gt[j]_class] += 1
        # 如果某个pred[i]负样本且没有哪个gt与之对应 那么[pred[i]_class nc] += 1
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes 数据集类别个数
        self.conf = conf # 预测框置信度阈值
        self.iou_thres = iou_thres # iou阈值 

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.  
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class =[300, 6] 一个batch中一张图的预测信息  其中x1y1x2y2是映射到原图img的
            labels (Array[M, 5]),  = [17, 5] class, x1, y1, x2, y2 其中x1y1x2y2是映射到原图img的
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return
        # [10, 6] 筛除置信度过低的预测框(和nms差不多)
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int() # 所有gt框类别(int) [17]  类别可能会重复
        detection_classes = detections[:, 5].int()  # 所有pred框类别(int) [10] 类别可能会重复  Positive + Negative
        iou = box_iou(labels[:, 1:], detections[:, :4]) # 求出所有gt框和所有pred框的iou [17, x1y1x2y2] + [10, x1y1x2y2] => [17, 10] [i, j] 第i个gt框和第j个pred的iou

        # iou > self.iou_thres: [17, 10] bool 符合条件True 不符合False
        # x[0]: [10] gt_index  x[1]: [10] pred_index   x合起来看就是第x[0]个gt框和第x[1]个pred的iou符合条件
        # 17 x 10个iou 经过iou阈值筛选后只有10个满足iou阈值条件
        x = torch.where(iou > self.iou_thres)
        # 后面会专门对这里一连串的matches变化给个实例再解释
        if x[0].shape[0]: # 存在大于阈值的iou时
             # torch.stack(x, 1): [10, gt_index+pred_index]
            # iou[x[0], x[1]][:, None]): [10, 1]   x[0]和x[1]的iou
            # 1、matches: [10, gt_index+pred_index+iou] = [10, 3]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # 2、matches按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 3、取第二列中各个框首次出现(不同预测的框)的行(即每一种预测的框中iou最大的那个)
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 4、matches再按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 5、取第一列中各个框首次出现(不同gt的框)的行(即每一种gt框中iou最大的那个)
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # 经过这样的处理 最终得到每一种预测框与所有gt框中iou最大的那个(在大于阈值的前提下)
                # 预测框唯一  gt框也唯一  这样得到的matches对应的Pred都是正样本Positive
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0  # 满足条件的iou是否大于0个 bool
            # a.transpose(): 转换维度 对二维数组就是转置 这里的matches: [9, gt_index+pred_index+iou] -> [gt_index+pred_index+iou, 9]
        # m0: [1, 9] 满足条件(正样本)的gt框index(不重复)  m1: [1, 9] 满足条件(正样本)的pred框index(不重复)
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        # 返回这个混淆矩阵
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        """
        :params normalize: 是否将混淆矩阵归一化 默认True
        :params save_dir: runs/train/expn 混淆矩阵保存地址
        :params names: 数据集的所有类别名
        :return None
        """
        import seaborn as sn # 图形展示库

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns  # 混淆矩阵归一化 0~1
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00) # 混淆矩阵中小于0.005的值被认为NaN

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)  # 初始化画布
        nc, nn = self.nc, len(names)  # number of classes, names  
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size 设置label的字体大小
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels 绘制混淆矩阵时 是否使用names作为labels
        ticklabels = (names + ['background']) if labels else "auto"
        with warnings.catch_warnings(): # 绘制热力图 即混淆矩阵可视化
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            # sean.heatmap: 热力图  data: 数据矩阵  annot: 为True时为每个单元格写入数据值 False用颜色深浅表示
            # annot_kws: 格子外框宽度  fmt: 添加注释时要使用的字符串格式代码 cmap: 指色彩颜色的选择
            # square: 是否是正方形  xticklabels、yticklabels: xy标签
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30, 
                       annot_kws={
                           "size": 8}, # 热力图内填充数字大小
                       cmap='Blues', # 热力图的颜色
                       fmt='.2f',  # 热力图内填充数值的格式控制
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_ylabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: 预测框
    :params box2: 预测框
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area 交
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area 并
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union # 交并比
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width 两个框的最小闭包区域的width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height 两个框的最小闭包区域的height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    
    """用于计算混淆矩阵
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    """在ComputeLoss类的build_targets函数中被调用(老版正样本筛选条件)
    :params wh1: anchors 当前feature map的3个anchor  [N, 2]
    :params wh2: t[:, 4:6] gt框的wh(没筛选 所有的gt)  [M, 2]
    :return 返回wh1和wh2的iou(矩阵)
    """
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    """用于ap_per_class函数
    Precision-recall curve  绘制PR曲线
    :params px: [1000] 横坐标 recall 值为0~1直接取1000个数
    :params py: list{nc} nc个[1000] 所有类别在IOU=0.5,横坐标为px(recall)时的precision
    :params ap: [nc, 10] 所有类别在每个IOU阈值下的平均mAP
    :params save_dir: runs\test\exp\PR_curve.png  PR曲线存储位置
    :params names: {dict:80} 数据集所有类别的字典 key:value
    """

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True) # 设置画布
    py = np.stack(py, axis=1)  # [1000, nc]

    # 画出所有类别在10个IOU阈值下的PR曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):  # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')# 设置x轴标签
    ax.set_ylabel('Precision')# 设置y轴标签
    ax.set_xlim(0, 1)# x=[0, 1]
    ax.set_ylim(0, 1)  # y=[0, 1]
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 显示图例
    ax.set_title('Precision-Recall Curve') 
    fig.savefig(save_dir, dpi=250) # 保存PR_curve.png图片
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    """用于ap_per_class函数
    Metric-Confidence curve 可用于绘制 F1-Confidence/P-Confidence/R-Confidence曲线
    :params px: [0, 1, 1000] 横坐标 0-1 1000个点 conf   [1000]
    :params py: 对每个类, 针对横坐标为conf=[0, 1, 1000] 对应的f1/p/r值 纵坐标 [71, 1000]
    :params save_dir: 图片保存地址
    :parmas names: 数据集names
    :params xlabel: x轴标签
    :params ylabel: y轴标签
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)  # 设置画布
    # 画出所有类别的F1-Confidence/P-Confidence/R-Confidence曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes 
        for i, y in enumerate(py):  # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else: # 如果>=21 classes 显示图例就会很乱 所以就不显示图例了 可以直接输入数组 x[1000] y[1000, 71]
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)
    # 画出所有类别在每个x点(conf)对应的均值F1-Confidence/P-Confidence/R-Confidence曲线
    y = smooth(py.mean(0), 0.05) #[1000] 求出所以类别在每个x点(conf)的平均值
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel) # 设置x轴标签
    ax.set_ylabel(ylabel) # 设置y轴标签
    ax.set_xlim(0, 1)  # x=[0, 1]
    ax.set_ylim(0, 1)  # y=[0, 1]
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left") # 显示图例
    ax.set_title(f'{ylabel}-Confidence Curve')  
    fig.savefig(save_dir, dpi=250) # 保存png图片
    plt.close(fig)
