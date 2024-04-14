import copy

from .GHM_package import GHM_module


class Loss_list_Manager:
    def __init__(self, threshold=0):
        self.maxinum_num = 150
        self.threshold = 0

    def load_loss(self, loss_list, loss_item):
        loss_item = loss_item.detach().item()
        if len(loss_list) >= self.maxinum_num:
            loss_list.pop(0)
        loss_list.append(loss_item)
        back_up_loss_list = copy.deepcopy(loss_list)
        back_up_loss_list.sort(reverse=True)
        if len(loss_list) >= self.maxinum_num:
            should_add = (back_up_loss_list.index(loss_item) > self.threshold)
        else:
            should_add = True

        return loss_list, should_add
