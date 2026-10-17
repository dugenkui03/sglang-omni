"""按任务独立选择素材；选中过的样本与请求是否成功无关。"""

import random

from tts_workbench.config import RunConfig
from tts_workbench.dataset import DialogueLine


class DialogueSampler:
    """在已筛选的非空素材内顺序循环，或按概率抽取已选/未选样本。"""

    def __init__(self, lines: list[DialogueLine], config: RunConfig) -> None:
        """为每个任务重建抽样状态，不继承其他任务的选择或模型缓存状态。"""
        self.lines: list[DialogueLine] = lines
        self.config: RunConfig = config
        self.unseen: list[DialogueLine] = list(lines)
        self.seen: list[DialogueLine] = []
        self.sequence: int = 0
        self.random: random.Random = random.Random()

    def next_line(self) -> DialogueLine:
        """先按概率选择样本池，再在池内等概率抽样；空池自动使用另一侧。"""
        if self.config.sample_order == "sequential":
            line = self.lines[self.sequence % len(self.lines)]
            self.sequence += 1
            return line
        elif self.seen and (
            not self.unseen or self.random.random() < self.config.repeat_probability
        ):
            return self.random.choice(self.seen)
        else:
            # 交换后移除，保持未选样本等概率且每次抽样开销恒定。
            index = self.random.randrange(len(self.unseen))
            line = self.unseen[index]
            self.unseen[index] = self.unseen[-1]
            self.unseen.pop()
            self.seen.append(line)
            return line
