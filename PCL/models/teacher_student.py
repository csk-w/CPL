# CCC/models/teacher_student.py

import torch
import torch.nn as nn
from . import create


class TeacherStudentModel(nn.Module):
    """
    封装了Student和Teacher网络的模型。
    - Student网络通过梯度进行训练。
    - Teacher网络不进行梯度更新，其权重是Student的指数移动平均（EMA）。
    """

    def __init__(self, arch, ema_alpha=0.999, **model_args):
        super(TeacherStudentModel, self).__init__()
        self.ema_alpha = ema_alpha

        self.student = create(arch, **model_args)
        self.teacher = create(arch, **model_args)

        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        self.teacher.eval()

    def forward(self, x):
        """
        默认的前向传播只通过Student网络。
        在训练循环中，我们将根据需要显式调用 self.teacher(x)。
        """
        return self.student(x)

    @torch.no_grad()
    def _update_teacher(self):
        """
        在每个训练迭代后，执行Teacher权重的EMA更新。
        """
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(self.ema_alpha).add_(param_s.data, alpha=1. - self.ema_alpha)