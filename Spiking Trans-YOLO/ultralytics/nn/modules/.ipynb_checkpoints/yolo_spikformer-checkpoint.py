# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import warnings
import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

# from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

# from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer,MSDeformAttn
from .utils import bias_init_with_prob, linear_init
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch
# from visualizer import get_local


from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math
# __all__ = ('MS_GetT','MS_CancelT', 'MS_ConvBlock','MS_Block','MS_DownSampling',
#            'MS_StandardConv','SpikeSPPF','SpikeConv','MS_Concat','SpikeDetect'
#            ,'Ann_ConvBlock','Ann_DownSampling','Ann_StandardConv','Ann_SPPF','MS_C2f',
#            'Conv_1','BasicBlock_1','BasicBlock_2','Concat_res2','Sample','MS_FullConvBlock','MS_ConvBlock_resnet50','MS_AllConvBlock','MS_ConvBlock_res2net')


decay = 0.25  # 0.25 # decay constants




class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value # 使用MultiSpike4，改变最大值
        # qtrick: 使用自定义的MultiSpike4类，定义了量化的脉冲发放机制，限制最大值为4。

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device) # 初始化脉冲
        output = torch.zeros_like(x) # 输出初始化为全零张量
        mem_old = 0 # 旧的膜电位初始化为0
        time_window = x.shape[0] # 时间窗口大小
        for i in range(time_window): # 迭代每个时间步
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i] # 根据衰减公式更新膜电位  """在 PyTorch 中，spike.detach() 是将 detach() 方法应用于一个名为 spike 的张量。这种方法会创建一个新的张量，它与原始张量共享相同的数据，但不会在计算图中跟踪其操作。"""

            else:
                mem = x[i]
            spike = self.qtrick(mem) # 使用自定义量化函数产生脉冲

            mem_old = mem.clone() # 复制当前的膜电位，用于下一个时间步
            output[i] = spike # 记录脉冲发放
        # print(output[0][0][0][0])
        return output # 返回脉冲发放序列

class MultiSpike4_2(nn.Module):
    # 自定义反向传播过程，控制输入的梯度在指定范围内传播。
    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input) # ctx.save_for_backward(input): 将input保存起来，供反向传播使用。ctx是一个上下文对象，用于存储反向传播时需要的变量。
            return torch.round(torch.clamp(input, min=-1, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < -1] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

"""
定义了量化脉冲的机制，将输入限制在0到4之间，用于模拟脉冲神经元的发放。
"""
class MultiSpike4(nn.Module):
    # 自定义反向传播过程，控制输入的梯度在指定范围内传播。
    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input) # ctx.save_for_backward(input): 将input保存起来，供反向传播使用。ctx是一个上下文对象，用于存储反向传播时需要的变量。
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike2(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant2(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)

class MultiSpike1(nn.Module):

    class quant1(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)

@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)

class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  #[0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) #这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  #本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = mem_update()


    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction # 强制重建网格
    export = False  # export mode # 导出模式标志
    shape = None # 输入张量的形状，用于推理时的形状校验
    anchors = torch.empty(0)  # init  # 初始化锚点
    strides = torch.empty(0)  # init  # 初始化步长

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes# 类别数量
        self.nl = len(ch)  # number of detection layers # 检测层的数量 ，等于传入的通道数长度
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x) # DFL 通道数，控制边框回归输出
        self.no = nc + self.reg_max * 4  # number of outputs per anchor # 每个锚点的输出数量
        self.stride = torch.zeros(self.nl)  # strides computed during build # 在构建过程中计算步长
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)# 用于回归的卷积层序列
        self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)# 用于分类的卷积层序列
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()# 边框回归的离散傅里叶层（DFL）

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape  # 官方注释：BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        # 我的注释：获取输入张量的形状，假设输入维度为 TBCHW, 通过对时间维度取均值来获取 BCHW 形状  # 输出 shape 为 BCHW
        for i in range(self.nl):# 通过 cv2 卷积层提取回归信息，cv3 提取分类信息，并在通道维度上拼接
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2) # # 输出格式为 TBC(回归+分类)HW
            x[i] = x[i].mean(0)  #官方注释：[2，144，32，684]  #这个地方有时候全是1.之后debug看看 # 我的注释：输出格式为 BC(回归+分类)HW
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        # # 将每个检测层的输出重新调整为 (B, no, -1) 的格式，并沿 anchor 维度拼接
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            ## 如果是导出模式，避免 TensorFlow 中的 FlexSplitV 操作
            box = x_cat[:, :self.reg_max * 4]# 提取边框回归部分，格式为 B(回归*4, anchors)
            cls = x_cat[:, self.reg_max * 4:]# 提取分类部分，格式为 B(类别数, anchors)
        else:
            # 切分回归部分和分类部分
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]# box: B(回归*4, anchors), cls: B(类别数, anchors)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # # 使用 dfl 进行边框转换，最终输出为边框位置，格式为 B(4, anchors)
        """
        B(no, anchors) 的含义是：

        B: 批次大小 (Batch size)，表示在一次前向传播过程中同时处理的输入样本数量。
        no: 每个锚点 (anchor) 的输出数量，具体是由类别数量和回归参数组成的。公式为 no = nc + reg_max * 4，其中 nc 是类别数量，reg_max * 4 是每个锚点的边框回归参数数量。
        anchors: 锚点数量 (anchors)，表示每个特征图上生成的预定义边框数量。这些锚点会根据图像的不同区域来预测物体。
        因此，B(no, anchors) 表示一个张量，其中：

        B   是批次维度，表示处理了多少张图片；
        no 是每个锚点需要预测的值的数量（包括类别预测和边框回归）；
        anchors 是特征图上生成的锚点数量，每个锚点对应一次预测。
        例如，如果批次大小为 8，类别数 nc 为 80，回归参数为 4（reg_max=4），特征图上有 100 个锚点，那么输出格式会是 8 x (80 + 4*4) x 100，即 8 x 96 x 100。
        
        
        """

        if self.export and self.format in ('tflite', 'edgetpu'):
            # 针对 TFLite 和 EdgeTPU 进行边框的归一化处理
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size
        # 将边框和分类概率拼接，返回格式为 B(回归4 + 分类数, anchors)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)# 导出模式返回最终预测，否则返回 (预测结果，原始回归和分类输出)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module # m = self：该行将 m 设为 self，也就是当前 SpikeDetect 实例。这使得后续代码可以通过 m 访问类的属性和方法。
        """
        注释掉的代码：注释的部分用于计算类别频率 cf 和类别频率的对数 ncf，可能是某些场景下初始化偏置的另一种方式。不过，这段代码在这里没有实际执行，只是留下了计算类别分布的参考方法。
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        """
        这里通过 for 循环迭代 m.cv2 和 m.cv3 中的卷积层，同时获取每一层对应的步长 s。
        这说明模型的多个卷积层（通常是回归和分类卷积层）需要逐层处理。cv2 通常处理回归任务（预测边框），cv3 处理分类任务（预测类别）。
        """
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box 这一行代码初始化回归卷积层的偏置项（a[-1].conv.bias），将其所有的偏置值都设为 1.0。a[-1] 访问的是 cv2 中最后一个卷积层，而该层的 conv.bias 是卷积层的偏置。这里的 1.0 的初始化值通常用于让边框回归层的初始输出值适中，避免一开始预测的边框过大或过小。
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            """
            这一行对分类卷积层 b[-1] 的偏置进行初始化。这里 b[-1].conv.bias.data[:m.nc] 代表分类任务的偏置项
            计算偏置的公式为 math.log(5 / m.nc / (640 / s) ** 2)，具体含义如下：
            m.nc：是类别数量。
            640：是默认图像尺寸（通常是 YOLO 模型的默认输入图像尺寸）。
            s：是当前层的步长。
            math.log(5 / m.nc / (640 / s) ** 2) 计算得到的值用来初始化分类层的偏置，确保网络的初始预测值较为合理。通过这种初始化方式，网络初始时对每个类别的预测会有一个较低的置信度（接近 0.01）。
            """


"""
直接从Meta中迁移的
用于处理 批归一化（Batch Normalization, BN） 和 填充（Padding）
"""
class BNAndPadLayer(nn.Module):
    """
    pad_pixels：整数，表示在特征图的每个边缘需要填充的像素数。
    num_features：整数，输入特征图的通道数，即批归一化层的通道数量。
    eps：浮点数，防止除以零的极小值，默认值为 1e-5。
    momentum：浮点数，用于更新运行时均值和方差的动量项，默认值为 0.1。
    affine：布尔值，表示是否学习可训练的缩放和平移参数，默认值为 True。
    track_running_stats：布尔值，表示是否跟踪运行时的均值和方差，默认值为 True。
    功能：
    调用父类 nn.Module 的初始化方法，确保模块正常初始化。
    创建一个 二维批归一化层 self.bn，用于对输入的特征图进行批归一化操作。
    保存需要填充的像素数 pad_pixels，用于后续的填充操作。

    """
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input): # 接收输入张量 input，形状为 [N, C, H, W]，
        output = self.bn(input)
        if self.pad_pixels > 0: #检查是否需要进行填充操作。如果 pad_pixels 大于 0，则进行填充。
            if self.bn.affine: # 检查批归一化层是否具有可学习的仿射参数（即缩放参数 weight 和偏置参数 bias）。
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            """
            使用 torch.nn.functional.pad 函数对批归一化后的输出张量进行填充。
            [self.pad_pixels] * 4 生成一个列表 [pad_pixels, pad_pixels, pad_pixels, pad_pixels]，表示在每个维度（左、右、上、下）填充 pad_pixels 个像素。
            填充后的张量形状变为 [N, C, H + 2 * pad_pixels, W + 2 * pad_pixels]。
            """
            output = F.pad(output, [self.pad_pixels] * 4)
            # 将计算得到的 pad_values 重塑为形状 [1, C, 1, 1]，以便在后续赋值时能够进行广播（broadcasting），即将相同的填充值应用到对应的通道上。
            """
            对填充后的输出张量的边缘进行赋值，将填充值 pad_values 赋给填充的像素区域。
            """
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output #返回经过批归一化和填充处理后的张量。

    """
    属性访问器
    为了方便访问批归一化层的参数，类中定义了以下属性方法：
    这些属性方法允许外部直接访问批归一化层 self.bn 的参数，如 weight、bias、running_mean、running_var 和 eps，方便在训练和调试过程中对这些参数进行检查或修改。
    """
    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

"""
RepConv 是一种通过 结构重参数化 的卷积层，通常用于在深度学习模型中优化推理效率。
它的设计目标是在 训练阶段 通过多个卷积分支增强模型的表现力，而在 推理阶段 将这些分支合并为一个单一的卷积层，以减少计算开销，加速推理过程。
通过这种方式，RepConv 可以同时兼顾训练时的高表现力和推理时的高效率。

"""
class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,# 加入卷积核尺寸，Meta中此处固定为3
            bias=False, # 取消bias' # 是否使用偏置，默认为False
            group = 1 # 卷积的分组数，默认1，表示标准卷积 # 加入group=1，在meta中下面用到的地方直接groups=1；
    ):
        super().__init__()
        # 计算填充大小，确保输出的特征图尺寸与输入一致
        padding = int((kernel_size-1)/2) # 加入了padding
        # hidden_channel = in_channel
        # 第一个卷积：1x1 卷积，用于保持输入的通道数不变（或用于分组卷积）
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        # 批归一化层，用于对卷积输出进行归一化，平衡不同通道的数值范围
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel) # 从pad_pixels=1变为pad_pixels=padding
        """
         # 定义卷积块，包括两个卷积层
        # 1. 3x3 深度可分离卷积（groups=in_channel 表示每个通道独立卷积）
        # 2. 1x1 卷积，减少通道数
        
        """
        conv3x3 = nn.Sequential(
            # mem_update(), #11111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
            # mem_update(),  #11111
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


# SpikeYOLO自己加的
class SepRepConv(nn.Module): #放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
#         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1,0, groups=group, bias=False),  #这里也是分组卷积
            # mem_update(), #11111
            nn.Conv2d(out_channel, out_channel, kernel_size,  1,0, groups=out_channel, bias=False),
        )


        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #Meta最后两个参数7,3，变成3,1
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2) # 加入padding
        med_channels = int(expansion_ratio * dim)
        # self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy") # 这句不要了
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  #这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
#         print("x.shape:",x.shape)
        x = self.lif1(x) #x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class DilatedConv(nn.Module):
    def __init__(self, input_dim):  # in_channels(out_channels), 内部扩张比例
        super().__init__()



        self.conv1 = nn.Conv2d(input_dim, input_dim, padding=1, kernel_size=3, groups=input_dim,
                                dilation=1)
        self.conv2 = nn.Conv2d(input_dim, input_dim, padding=2, kernel_size=3, groups=input_dim,
                                dilation=2)
        self.conv3 = nn.Conv2d(input_dim, input_dim, padding=3, kernel_size=3, groups=input_dim,
                                dilation=3)
        self.conv4 = nn.Conv2d(input_dim, input_dim, padding=4, kernel_size=3, groups=input_dim,
                                dilation=4)

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)
        self.bn3 = nn.BatchNorm2d(input_dim)
        self.bn4 = nn.BatchNorm2d(input_dim)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()
        self.lif4 = mem_update()
        self.liffuse = mem_update()

        self.fuse = nn.Sequential(
            nn.Conv2d(input_dim * 4, input_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(input_dim),
        )

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape

        c1 = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, C, H, W)
        c2 = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        c3 = self.bn3(self.conv3(self.lif3(x).flatten(0, 1))).reshape(T, B, C, H, W)
        c4 = self.bn4(self.conv4(self.lif4(x).flatten(0, 1))).reshape(T, B, C, H, W)

        cat = torch.cat([c1, c2, c3, c4], dim=2)
        fuse = self.fuse(self.liffuse(cat).flatten(0, 1)).reshape(T, B, C, H, W)

        return fuse


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.dila = DilatedConv(input_dim)

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        
        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) #137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进




    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(self.dila(x)) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
            #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

    
class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)



    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        #print(x.shape)
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x

"""
forward 方法
T, B, _, _, _ = x.shape: 获取输入张量的形状。这里 T 是时间步长，B 是批次大小，_ 表示输入张量的通道数、高度和宽度。

if hasattr(self, "encode_lif"):: 检查是否存在脉冲神经元更新机制（即 self.encode_lif 是否被定义）。如果存在，则使用脉冲神经元的膜电位更新机制 self.encode_lif 处理输入数据。这表示当层不是网络的第一层时，输入数据会通过脉冲神经元的膜电位更新。

x = self.encode_conv(x.flatten(0, 1)): 将输入数据在第0和第1维展平（将时间步和批次维度合并），形成一个 [B * T, C, H, W] 形状的张量，然后应用卷积操作。卷积操作会对输入进行下采样，即减少输入特征图的空间维度（高度和宽度）。

_, C, H, W = x.shape: 获取卷积后的输出形状，C 是卷积后的通道数，H 和 W 是卷积后的高度和宽度。

x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous(): 将卷积后的输出通过批归一化层处理，并将张量的形状恢复为 [T, B, C, H, W]（即时间步、批次、通道数、高度和宽度），其中 contiguous() 确保张量在内存中的布局是连续的。
"""
class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        # 定义卷积层，kernel_size 指定卷积核大小，stride 为步幅，padding 为填充
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        # 定义批归一化层，规范化卷积输出，保持训练的稳定性
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        # 如果不是第一层，使用脉冲神经元模型更新膜电位
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        # 如果存在脉冲神经元更新机制（即不是第一层），则通过脉冲神经元处理输入数据
        if hasattr(self, "encode_lif"): #如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)
        """这里就是先脉冲在卷积，就是脉冲卷积操作了"""
        # 使用卷积层进行下采样（flatten 0 和 1 维表示将时间步与批次维度展平，卷积操作需要 [B * T, C, H, W] 形式）
        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x

"""
MS_GetT 是一个非常简单的 PyTorch 模块，主要功能是将输入的数据扩展到时间维度（Time Dimension）。
在脉冲神经网络（Spiking Neural Networks, SNN）或序列数据处理中，常常需要引入时间维度来表示脉冲神经元在不同时间步的动态变化。
这个类的作用是将输入的数据沿时间维度复制多次，生成一个时间序列。
"""
class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T# 时间步长 T，表示数据在时间维度上复制的次数
        self.in_channels = in_channels

    def forward(self, x):
        # x.unsqueeze(0) 在输入张量 x 的第0维度上增加一个维度，代表时间维度。
        # repeat(self.T, 1, 1, 1, 1) 在时间维度上重复输入 T 次
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x # 返回复制后的张量

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x

class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # 标准卷积，并结合了脉冲神经元更新机制
    default_act = nn.SiLU()  # default activation # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 定义卷积层，使用自动填充，groups 和 dilation 参数允许调整卷积行为
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 使用脉冲神经元的膜电位更新机制
        self.lif = mem_update()
        # 批归一化
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            res = self.cv2(torch.cat((x, y1, y2, y3), 2))
            # print("x: ",x.shape)
            # print("y1: ",y1.shape)
            # print("y2: ",y2.shape)
            # print("y3: ",y3.shape)
            # print("res: ",res.shape)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))



class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)


# _______________________以下是自己加入的______________________________________
# 效果不行，先废弃，用本来的
class Spike_RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.


    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.
    这个类实现了实时可变形变换器解码器（RTDETRDecoder），用于物体检测。
    它使用变换器（Transformer）架构以及可变形卷积，来预测图像中物体的边界框和类别标签。
    该模块整合了来自多个层的特征，通过多层 Transformer 解码器输出最终预测结果。
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,# 类别数量，默认为 80
            ch=(512, 1024, 2048),# Backbone 特征图的通道数
            hd=256,  # hidden dim # 隐藏层维度
            nq=300,  # num queries # 查询点数量
            ndp=4,  # num decoder points # 解码器每层点的数量
            nh=8,  # num head # 多头注意力机制的头数
            ndl=6,  # num decoder layers # 解码器层数
            d_ffn=1024,  # dim of feedforward # Feedforward 网络的维度
            dropout=0., # Dropout 概率，默认为 0
            act=nn.ReLU(), # 激活函数，默认是 ReLU
            eval_idx=-1, # 评估索引，默认为 -1
            # Training args
            nd=100,  # num denoising# Denoising 训练中的噪声数量
            label_noise_ratio=0.5, # 标签噪声比率
            box_noise_scale=1.0, # 边界框噪声比例
            learnt_init_query=False): # 是否学习初始查询嵌入
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.

            初始化 RTDETRDecoder 模块并传入参数。

        Args:
            nc (int): 类别数量，默认是 80。
            ch (tuple): Backbone 特征图的通道数，默认为 (512, 1024, 2048)。
            hd (int): 隐藏层维度，默认是 256。
            nq (int): 查询点数量，默认是 300。
            ndp (int): 解码器每层点的数量，默认是 4。
            nh (int): 多头注意力机制的头数，默认是 8。
            ndl (int): 解码器层数，默认是 6。
            d_ffn (int): Feedforward 网络的维度，默认是 1024。
            dropout (float): Dropout 概率，默认是 0。
            act (nn.Module): 激活函数，默认是 ReLU。
            eval_idx (int): 评估索引，默认是 -1。
            nd (int): Denoising 中的噪声数量，默认是 100。
            label_noise_ratio (float): 标签噪声比率，默认是 0.5。
            box_noise_scale (float): 边界框噪声比例，默认是 1.0。
            learnt_init_query (bool): 是否学习初始查询嵌入，默认是 False。
        """
        super().__init__() # 调用父类的初始化方法
        self.hidden_dim = hd # 隐藏层的维度
        self.nhead = nh # 注意力机制的头数
        self.nl = len(ch)  # num level # 检测层的数量，等于传入的通道数长度
        self.nc = nc# 类别数
        self.num_queries = nq # 查询点的数量
        self.num_decoder_layers = ndl # 解码器层数

        # Backbone feature projection
        #self.input_proj = nn.ModuleList(nn.Sequential(SpikeConv(x, hd, 1), nn.BatchNorm2d(hd)) for x in ch)  # 1x1 卷积将特征图通道数转为隐藏层维度 # 归一化处理


        #self.input_proj = nn.ModuleList(SpikeConv(x, hd, 1) for x in ch)  # 错误，废弃
        self.conv= nn.ModuleList(SpikeConv(x, hd, 1) for x in ch)


        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module

        decoder_layer = MS_DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp) # 定义一个可变形 Transformer 解码器层
        self.decoder = MS_DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx) # 创建完整的解码器

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)# 嵌入层，生成 denoising 的类别嵌入
        self.num_denoising = nd # denoising 的数量
        self.label_noise_ratio = label_noise_ratio# 标签噪声比率
        self.box_noise_scale = box_noise_scale# 边界框噪声缩放比例

        # Decoder embedding # 解码器嵌入部分
        self.learnt_init_query = learnt_init_query # 是否学习初始查询嵌入
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)# 查询点的嵌入
        # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)# MLP 用于查询点的坐标嵌入
        self.query_pos_head = MS_MLP(4, 2 * hd, hd)

        # Encoder head# 编码器头
        #self.enc_output = nn.Sequential(SpikeLinear(hd, hd), nn.BatchNorm1d(hd))# 全连接层# LayerNorm
        self.enc_output = SpikeLinearWithBN(hd, hd)
        self.enc_score_head = SpikeLinear(hd, nc)# 用于预测类别的线性层
        self.enc_bbox_head = MS_MLP(hd, hd, 4)# 用于预测边界框的 MLP 层

        # Decoder head解码器头
        self.dec_score_head = nn.ModuleList([SpikeLinear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MS_MLP(hd, hd, 4) for _ in range(ndl)])

        #self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        """
                前向传播，返回边界框和分类得分。
                """
        from ultralytics.models.utils.ops import get_cdn_group



        # Input projection and embedding# 输入投影和嵌入
        feats, shapes = self._get_encoder_input(x)# 获取投影特征和形状

        # Prepare denoising training准备 denoising 训练
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training) # 准备 denoising 嵌入和相关参数
        # 获取解码器的输入
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        """
        输入：
        embed：形状 [B, num_queries, C]，解码器的嵌入输入。
        refer_bbox：形状 [B, num_queries, 4]。
        feats：编码后的特征 [B, H*W, C]。
        经过解码器：
        dec_bboxes：解码器输出的边界框，形状为 [B, num_queries, 4]。
        dec_scores：分类得分，形状为 [B, num_queries, nc]（nc 是类别数量）。
        """
        # Decoder # 解码器部分
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask) # 通过解码器获得边界框和分类得分
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta # 输出边界框和分类结果
        #print("dec_bboxes: ", dec_bboxes.shape)
       # print("dec_scores: ", dec_scores.shape)
        if self.training:
            return x # 如果是训练模式，返回输出
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1) # 将边界框和分类结果拼接
        return y if self.export else (y, x)# 如果是导出模式，返回拼接后的结果；否则返回原始结果

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        """
                生成给定形状和特定网格大小的锚框，并进行验证。
                """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)#  # 生成 y 轴的坐标索引 (0 到 h-1)
            sx = torch.arange(end=w, dtype=dtype, device=device)# # 生成 x 轴的坐标索引 (0 到 w-1)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx) # 根据 PyTorch 版本选择 meshgrid 的方式，生成网格 (grid_x, grid_y)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)  # 将 x 和 y 坐标堆叠为 (h, w, 2)，表示每个位置的网格坐标 (x, y)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# # 创建用于归一化的宽高 tensor，将 (h, w) 存储为张量用于后续的计算
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)# 将网格坐标归一化到 [0, 1] 范围，并加 0.5 使其中心对齐到网格单元中心 # (1, h, w, 2)，增加维度以支持后续批处理
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)# 根据当前层级计算网格大小，生成宽高 (wh) 张量 # 每一层的锚框大小随层级指数增长
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4) # 将 (x, y) 坐标和 (width, height) 合并为 (1, h*w, 4)，代表锚框的四个参数

        # 将所有层的锚框拼接在一起，形成最终的锚框集合
        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1 # 创建有效锚框的掩码，检查锚框是否都在 (eps, 1 - eps) 的范围内
        anchors = torch.log(anchors / (1 - anchors)) # 使用对数几率函数将锚框映射到无限范围，用于后续损失计算等处理
        anchors = anchors.masked_fill(~valid_mask, float('inf'))  # 将无效的锚框（掩码为 False 的部分）填充为正无穷大，避免后续被使用
        return anchors, valid_mask # 返回生成的锚框和有效锚框的掩码

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them.
        # 处理输入并返回编码器输入，通过从输入中获取投影特征并进行拼接。"""
        # Get projection features
        # x = [self.input_proj[i](feat) for i, feat in enumerate(x)]

        for i, feat in enumerate(x):
            #print(feat.shape)
            x[i] = self.conv[i](feat)
            x[i] = x[i].mean(0)


        # Get encoder inputs
        feats = [] # 初始化列表，用于存储处理后的特征
        shapes = []# 初始化列表，用于存储每个特征图的形状
        # 遍历每个投影后的特征
        for feat in x:
            h, w = feat.shape[2:] # 获取特征图的高度和宽度（h 和 w）
            # [b, c, h, w] -> [b, h*w, c] # 将特征图从形状 [b, c, h, w] 转换为 [b, h*w, c]，以便于后续处理
            feats.append(feat.flatten(2).permute(0, 2, 1))# flatten(2) 将前两维展平，permute(0, 2, 1) 改变维度顺序
            # [nl, 2] # 存储特征图的形状 [h, w]
            shapes.append([h, w])

        # [b, h*w, c] # 将所有特征在第一维进行拼接，形成最终的特征表示
        feats = torch.cat(feats, 1) # [b, h*w, c]，将所有特征图在高度维度拼接在一起
        return feats, shapes # 返回处理后的特征和每个特征图的形状

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        # 从提供的特征和形状生成并准备解码器所需的输入。
        bs = len(feats) # 获取批次大小，即特征的数量 feats此时的维度为[b, h*w, c]
        # Prepare input for decoder  # 为解码器准备输入
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device) # 生成锚框和有效掩码，锚框的形状基于特征的形状，并保持与输入特征相同的数据类型和设备
        #print((valid_mask * feats).shape)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 # 使用有效掩码处理特征，得到编码器输出的特征

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)  # 通过评分头获取编码器输出特征的分数

        # Query selection # 查询选择
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1) # 获取分数最高的 num_queries 个查询的索引，并展平为一维
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1) # 创建一个包含批次索引的张量，重复以匹配查询的数量

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) # 选择出对应于最高分查询的特征，并重塑为 [bs, num_queries, 256] 形状
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1) # 选择与最高分查询对应的锚框，并重塑为 [bs, num_queries, 4] 形状

        # Dynamic anchors + static content  # 动态锚框 + 静态内容
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors # 计算参考边界框：将解码器输出的边界框与锚框相加
        """
        refer_bbox 作为解码器的锚框，形状为 [B, num_queries, 4]，即每个查询点预测的边界框坐标。
        """

        enc_bboxes = refer_bbox.sigmoid() # 对参考边界框应用 sigmoid 函数以获得边界框的坐标
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # 如果存在 dn_bbox，将其与 refer_bbox 拼接
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # 获取与选择的查询对应的得分，并重塑为 [bs, num_queries, -1] 形状

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        # 如果学习初始化查询，则复制嵌入权重；否则使用选择的特征作为嵌入
        if self.training:
            refer_bbox = refer_bbox.detach() # 在训练期间，从计算图中分离参考边界框
            if not self.learnt_init_query:
                embeddings = embeddings.detach() # 如果不是学习初始化查询，则也分离嵌入
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # 如果存在 dn_embed，将其与 embeddings 拼接

        return embeddings, refer_bbox, enc_bboxes, enc_scores # 返回解码器的输入：嵌入、参考边界框、编码边界框和编码分数

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

"""
新的，只改了最初的输入，后面的所有注意力都没有改
"""
class SpikeRTDETRDecoderv1(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        # self.input_proj = nn.ModuleList(nn.Sequential(SpikeConvWithoutBN(x, hd, 1), nn.BatchNorm2d(hd)) for x in ch)
        self.input_proj = nn.ModuleList(SpikeConv(x, hd, 1) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        #print("dec_bboxes: ", dec_bboxes.shape)
        #print("dec_scores: ", dec_scores.shape)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        for i, feat in enumerate(x):
            #print(feat.shape)
            x[i] = self.input_proj[i](feat)
            x[i] = x[i].mean(0)

        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer.conv.weight)

class SpikeRTDETRDecoderv2(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        # self.input_proj = nn.ModuleList(nn.Sequential(SpikeConvWithoutBN(x, hd, 1), nn.BatchNorm2d(hd)) for x in ch)
        self.input_proj = nn.ModuleList(SpikeConv(x, hd, 1) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = MS_DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = MS_DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MS_MLP(4, 2 * hd, hd)

        # Encoder head
        self.enc_output = nn.Sequential(SpikeLinear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = SpikeLinear(hd, nc)
        self.enc_bbox_head = MS_MLP(hd, hd, 4)

        # Decoder head
        self.dec_score_head = nn.ModuleList([SpikeLinear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MS_MLP(hd, hd, 4) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        #print("dec_bboxes: ", dec_bboxes.shape)
        #print("dec_scores: ", dec_scores.shape)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        for i, feat in enumerate(x):
            #print(feat.shape)
            x[i] = self.input_proj[i](feat)
            x[i] = x[i].mean(0)

        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.linear.bias, bias_cls)
        constant_(self.enc_bbox_head.fc1_conv.weight, 0.0)
        constant_(self.enc_bbox_head.fc1_conv.bias, 0.0)
        constant_(self.enc_bbox_head.fc1_bn.weight, 0.0)
        constant_(self.enc_bbox_head.fc1_bn.bias, 0.0)
        constant_(self.enc_bbox_head.fc2_conv.weight, 0.0)
        constant_(self.enc_bbox_head.fc2_conv.bias, 0.0)
        constant_(self.enc_bbox_head.fc2_bn.weight, 0.0)
        constant_(self.enc_bbox_head.fc2_bn.bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.linear.bias, bias_cls)
            constant_(reg_.fc1_conv.weight, 0.0)
            constant_(reg_.fc1_conv.bias, 0.0)
            constant_(reg_.fc1_bn.weight, 0.0)
            constant_(reg_.fc1_bn.bias, 0.0)
            constant_(reg_.fc2_conv.weight, 0.0)
            constant_(reg_.fc2_conv.bias, 0.0)
            constant_(reg_.fc2_bn.weight, 0.0)
            constant_(reg_.fc2_bn.bias, 0.0)

        linear_init(self.enc_output[0].linear)
        xavier_uniform_(self.enc_output[0].linear.weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.linear.weight)
        xavier_uniform_(self.query_pos_head.fc1_conv.weight)
        #xavier_uniform_(self.query_pos_head.fc1_bn.weight)
        xavier_uniform_(self.query_pos_head.fc2_conv.weight)
        #xavier_uniform_(self.query_pos_head.fc2_bn.weight)

        #xavier_uniform_(self.query_pos_head.layers[1].linear.weight)
        for layer in self.input_proj:
            xavier_uniform_(layer.conv.weight)

class SpikeRTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        # self.input_proj = nn.ModuleList(nn.Sequential(SpikeConvWithoutBN(x, hd, 1), nn.BatchNorm2d(hd)) for x in ch)
        self.input_proj = nn.ModuleList(SpikeConv(x, hd, 1) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        #print("dec_bboxes: ", dec_bboxes.shape)
        #print("dec_scores: ", dec_scores.shape)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        for i, feat in enumerate(x):
            #print(feat.shape)
            x[i] = self.input_proj[i](feat)
            x[i] = x[i].mean(0)

        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer.conv.weight)

class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = DSSA_Head(d_model, n_heads,2,2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(1, 2), k.transpose(1, 2), embed.transpose(1, 2)).transpose(1, 2)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)



class SpikeLinear(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # 标准卷积，并结合了脉冲神经元更新机制
    default_act = nn.SiLU()  # default activation # 默认激活函数

    def __init__(self, in_features, out_features, bias=True,k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 定义卷积层，使用自动填充，groups 和 dilation 参数允许调整卷积行为
        self.linear = nn.Linear(in_features, out_features)
        # 使用脉冲神经元的膜电位更新机制
        self.lif = mem_updateDetect()
        # 批归一化
        #self.bn = nn.BatchNorm2d(out_features)
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.lif(x)
        x = self.linear(x)
        return x
class MS_MLP(nn.Module):
    """
    参数:
    in_features: 输入特征的数量。
    hidden_features: 隐藏层特征的数量（如果未提供，则默认为 in_features）。
    out_features: 输出特征的数量（如果未提供，则默认为 in_features）。
    drop: Dropout 率（代码中设置为 0.0，但实际上未使用 dropout）。
    layer: 可能用于控制特定层的标志（代码中未使用）。
    网络层:
    fc1_conv: 一个 1D 卷积层，作用于输入特征并生成隐藏特征。
    fc1_bn: 第一个卷积输出的批归一化层。
    fc1_lif: 脉冲神经网络组件，使用多步的 Leaky Integrate-and-Fire (LIF) 神经元模型，包含一个衰减常数 tau，并具有重置行为。
    fc2_conv: 一个 1D 卷积层，作用于隐藏特征并生成输出特征。
    fc2_bn: 第二个卷积输出的批归一化层。
    fc2_lif: 输出层的另一个 LIF 节点，用于模拟神经元的脉冲活动。
    """
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = mem_updateDetect()

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = mem_updateDetect()
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    """
    输入:

    x: 形状为 (T, B, C, H, W) 的输入张量，其中：
    T: 时间步长
    B: 批次大小
    C: 通道数
    H: 高度
    W: 宽度
    在前向传播过程中，输入张量会首先通过 fc1_lif 节点，模拟神经元的脉冲活动，接着通过 fc1_conv 卷积层和批归一化处理，最后通过 fc2_lif 和 fc2_conv 层得到输出。整体的流程包括特征的卷积、批归一化以及脉冲神经元模型的处理。

    主要逻辑:
    输入张量首先被压平，以便在卷积操作中使用，然后通过两个 LIF 节点和相应的卷积操作。
    每个卷积操作后进行批归一化以稳定训练过程。
    输出张量恢复为原始的输入维度 (T, B, C, H, W)。
    这个模块结合了卷积层和脉冲神经网络节点，适合用于时间序列或具有脉冲神经元特性的任务。
    """
    def forward(self, x):
        # T, B, C, H, W = x.shape
        #print("1:",x.shape)
        B,N,C=x.shape
        x = x.permute(0, 2, 1) # BCN
        #N = H * W
        #x = x.flatten(3)
        x = self.fc1_lif(x)
        #print("2:", x.shape)
        x = self.fc1_conv(x)
        #print("3:", x.shape)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        #print("4:", x.shape)
        x = self.fc2_lif(x)
        #print("5:", x.shape)
        x = self.fc2_conv(x)
        #print("6:", x.shape)
        x = self.fc2_bn(x)
        #print("7:", x.shape)
        x=x.reshape(B,self.c_output,N).contiguous()
        x = x.permute(0, 2, 1) # BNC


        return x

class MS_DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = DSSA_Head(d_model, n_heads,2,2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = SpikeLinear(d_model, d_ffn)
        self.act = mem_updateDetect()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = SpikeLinear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(1, 2), k.transpose(1, 2), embed.transpose(1, 2)).transpose(1, 2)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)

class MS_DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
class MS_MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = SpikeLinear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = SpikeLinear(d_model, n_heads * n_levels * n_points)
        self.value_proj = SpikeLinear(d_model, d_model)
        self.output_proj = SpikeLinear(d_model, d_model)

        self.lif = mem_updateDetect()

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.linear.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.linear.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.linear.weight.data, 0.)
        constant_(self.attention_weights.linear.bias.data, 0.)
        xavier_uniform_(self.value_proj.linear.weight.data)
        constant_(self.value_proj.linear.bias.data, 0.)
        xavier_uniform_(self.output_proj.linear.weight.data)
        constant_(self.output_proj.linear.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multi-scale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')


        value=self.lif(value)
        #value_shapes=self.lif(value_shapes)
        sampling_locations=self.lif(sampling_locations)
        attention_weights=self.lif(attention_weights)

        output = MS_multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)



def MS_multi_scale_deformable_attn_pytorch(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()

D = 4
class DSSA_Head(nn.Module):
    """


    DSSA的逻辑是先QKt
    在和V乘积


    """
    def __init__(self, dim,num_heads, lenth_patch, patch_size):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim # 输入通道
        self.num_heads = num_heads # 检测头
        self.lenth = lenth_patch
        self.patch_size=patch_size
        self.register_buffer('firing_rate_x', torch.zeros(1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = mem_updateDetect(act=False)

        # self.W = layer.Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode='m')
        self.W = SpikeConvDetect_WithoutBN(dim, dim, 1, 1)
        self.norm =  nn.BatchNorm1d(dim)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = mem_updateDetect(act=False)
        self.activation_out = mem_updateDetect(act=False)

        self.Wproj = SpikeConvDetect_WithoutBN(dim, dim,1,1)
        self.norm_proj =  nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor,k: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
        """
        这里的Q,K,V都是BNC维度，但是传进来时改变了形状，成为 BCN
        源代码是适用于BCN的
        原来的x是q，y1是K，y2是V
        """

        #print("DSSA_forward_x:",x.shape)
        # X: [T, B, C, H, W]
        #T, B, C, H, W = x.shape
        B, C, N = x.shape

        #print("x/q: ", x.shape)
        #print("k: ", k.shape)
        #print("v: ", v.shape)

        self.lenth= N // (self.patch_size**2)

        # shortcut最后加和
        v_feat = v.clone()

        x = self.W(x)  # B C N
        x = self.norm(x)
        x = self.activation_in(x) # B C N

        # y = self.W(x)
        # y = self.norm(y)
        """ k v 都是从参数得到的，不需要再从x变换获得"""
        k = self.W(k) # B C N
        k=self.norm(k)
        v = self.W(v) # B C N
        v=self.norm(v)


        """
        x y1 y2 [ T B nums_head,c/nums_head,N]
        """
        # y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        # y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        # x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)
        # print("DSSA_forward_y1:", y1.shape)
        # print("DSSA_forward_y2:", y2.shape)
        # print("DSSA_forward_xRE:", x.shape)

        """
        x k v [ B nums_head,c/nums_head,N]
        """
        # print("DSSA_forward_x:", x.shape)
        # print("DSSA_forward_k:", k.shape)
        # print("DSSA_forward_v:", v.shape)
        x = x.reshape(B, self.num_heads, C // self.num_heads, -1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            # 它的作用是创建一个新的 Tensor，这个新的 Tensor 和原始的 Tensor 共享相同的存储空间，但是和计算图断开连接，不再参与梯度计算。换句话说，detach()函数可以用来获取一个 Tensor 的副本，但是该副本不再与计算图相关联，因此它不会影响到反向传播过程。这个方法在需要将某个 Tensor 作为新的输入传递给某个函数或者模型时很有用，
            firing_rate_x = x.detach().mean((0, 2, 3), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                1 - self.momentum)
        #scale1 = 1. / torch.sqrt((4*self.firing_rate_x+12*self.firing_rate_x*self.firing_rate_x) * (self.dim // self.num_heads))
        #scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        scale1 = 1. / torch.sqrt((D*self.firing_rate_x+(D*(D-1))*self.firing_rate_x*self.firing_rate_x) * (self.dim // self.num_heads))
        attn = self.matmul1(k.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)
        #print("attn: ",attn.shape)
        if self.training:
            firing_rate_attn = attn.detach().mean((0, 2, 3), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (
                1 - self.momentum)
        #scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)
        #scale2 = 1. / torch.sqrt((4*self.firing_rate_attn+12*self.firing_rate_attn*self.firing_rate_attn) * self.lenth)
        scale2 = 1. / torch.sqrt((D * self.firing_rate_attn + D*(D+1) * self.firing_rate_attn * self.firing_rate_attn) * self.lenth)
        #print("DSSA_forward_attn:", attn.shape)
        #print("DSSA_forward_y2:", y2.shape)
        out = self.matmul2(v, attn)
        out = out * scale2
        # out = out.reshape(T, B, C, H, W)
        out = out.reshape(B, C, N)
        out = self.activation_out(out)
        #print("DSSA_forward_out:", out.shape)
        out = self.Wproj(out)
        out = self.norm_proj(out)
        #print("DSSA_forward_out:", out.shape)
        out = out + v_feat
        # (out.shape)
        return out
class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)
class mem_updateDetect(nn.Module):
    def __init__(self, act=False):
        super(mem_updateDetect, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value # 使用MultiSpike4，改变最大值
        # qtrick: 使用自定义的MultiSpike4类，定义了量化的脉冲发放机制，限制最大值为4。

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device) # 初始化脉冲
        output = torch.zeros_like(x) # 输出初始化为全零张量
        mem_old = 0 # 旧的膜电位初始化为0
        time_window = 1 # 时间窗口大小
        for i in range(time_window): # 迭代每个时间步
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i] # 根据衰减公式更新膜电位  """在 PyTorch 中，spike.detach() 是将 detach() 方法应用于一个名为 spike 的张量。这种方法会创建一个新的张量，它与原始张量共享相同的数据，但不会在计算图中跟踪其操作。"""

            else:
                mem = x[i]
            spike = self.qtrick(mem) # 使用自定义量化函数产生脉冲

            mem_old = mem.clone() # 复制当前的膜电位，用于下一个时间步
            output[i] = spike # 记录脉冲发放
        # print(output[0][0][0][0])
        return output # 返回脉冲发放序列

class SpikeConvDetect_WithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # 标准卷积，并结合了脉冲神经元更新机制
    default_act = nn.SiLU()  # default activation # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 定义卷积层，使用自动填充，groups 和 dilation 参数允许调整卷积行为
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 使用脉冲神经元的膜电位更新机制
        self.lif = mem_updateDetect()
        # 批归一化
        #self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # T, B, C, H, W = x.shape
        #H_new = int(H / self.s)
        #W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x)
        return x

# --------------------------------------- AIFI ----------------------------
class MS_TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        #self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model

        self.ma_S = DSSA_Head(c1, num_heads, 2, 2)

        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

        #self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        # if self.training:
        #     src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        #     #print(src2.shape)
        # else:
        #     src2 = self.ma_S(q, k, src)
        #     src2 = src2.permute(0, 2, 1)
        src2 = self.ma_S(q.permute(0, 2, 1), k.permute(0, 2, 1), src.permute(0, 2, 1))
        src2 = src2.permute(0, 2, 1)
        src2 = self.lif1(src2) # 自己加的
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.lif2(src)  # 自己加的
        src2 = self.fc2(self.dropout(self.lif3(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma_S(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class MS_AIFI(MS_TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        #c, h, w = x.shape[1:]
        T, B, c, h, w = x.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c) # 这里使用了2D正弦-余弦函数生成位置嵌入，适用于处理图像数据。
        x=x.flatten(0,1)
        # Flatten [B, C, H, W] to [B, HxW, C]
        #print("x: ", x.shape)
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([T, B, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]



