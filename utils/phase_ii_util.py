import multiprocessing
from types import FunctionType
import torch
import torch.nn.functional as F


class GradCam(object):
    def __init__(self, model, vis_layer_names, detach=True):
        self.model = model
        self.init_vis_layers(detach)

    def _add_hook(self, net, detach):
        if hasattr(net, '_modules'):
            for m_name, module in net._modules.items():
                if m_name in self.vis_layer_names:
                    if detach:
                        save_output_code = compile('def save_output' + m_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + m_name + '\"]["output"].append(output.detach());', "<string>", "exec")
                    else:
                        save_output_code = compile('def save_output' + m_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + m_name + '\"]["output"].append(output);', "<string>", "exec")
                    func_space = {'self': self}
                    func_space.update(globals())
                    save_output = FunctionType(
                        save_output_code.co_consts[0], func_space, "save_output")
                    h = module.register_forward_hook(save_output)
                    self.forward_hook_handles.append(h)

                    save_gradient_code = compile(
                        'def save_gradient' + m_name +
                        '(module, input_grad, output_grad): '
                        'vis_info = getattr(self, "vis_info");'
                        'vis_info[\"' + m_name + '\"]["grad"].append(output_grad[0]);', "<string>", "exec")
                    save_gradient = FunctionType(
                        save_gradient_code.co_consts[0], func_space, "save_gradient")
                    h = module.register_backward_hook(save_gradient)
                    self.backward_hook_handles.append(h)
                self._add_hook(module, detach)

    def add_hook(self, detach):
        for net_name in self.model.net_names:
            net = getattr(self.model, 'net_' + net_name)
            self._add_hook(net, detach)

    def init_vis_layers(self, detach):
        self.vis_info = {}
        for m_name in self.vis_layer_names:
            self.vis_info[m_name] = {}
            self.vis_info[m_name]['output'] = []
            self.vis_info[m_name]['grad'] = []

        self.forward_hook_handles = []
        self.backward_hook_handles = []
        self.add_hook(detach)

    def remove_hook(self):
        for h in self.forward_hook_handles:
            h.remove()
        self.forward_hook_handles = []
        for h in self.backward_hook_handles:
            h.remove()
        self.backward_hook_handles = []

    def cal_grad(self, y, t_label, retain_graph=False, create_graph=False):
        """
        Args:
            model:
            imgs: NHWC
            t_label: target label to be visualized
        """
        model = self.model
        output = y
        one_hots = torch.zeros(output.shape[0], output.shape[1]).cuda(
            model.opt.gpu_ids[0])
        one_hots[:, t_label] = 1
        ys = torch.sum(one_hots * output)
        ys.backward(retain_graph=retain_graph, create_graph=create_graph)

    def cal_cam(self):
        self.cat_info()
        result = dict()
        for key in self.vis_info.keys():
            grads_val = self.vis_info[key]['grad'] # (batch, 2048, 16, 16)
            feature = self.vis_info[key]['output'] # (batch, 2048, 16, 16)
            weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)
            cam = weights * feature
            cam = torch.sum(cam, dim=1)
            cam = F.relu(cam) # (batch, 16, 16)
            # normalize to (0, 1)
            tmp = cam.view(cam.shape[0], -1)
            max_value = torch.max(tmp, dim=1)[0] # (batch)
            max_value = max_value.unsqueeze(dim=1).unsqueeze(dim=1) # (batch, 1, 1)
            min_value = torch.min(tmp, dim=1)[0] # (batch)
            min_value = min_value.unsqueeze(dim=1).unsqueeze(dim=1) # (batch, 1, 1)
            cam.sub_(min_value).div_(max_value-min_value)
            # save to vis_info
            result[key] = cam.cpu().detach()
        return result

    def reset_info(self):
        for m_name in self.vis_layer_names:
            self.vis_info[m_name] = {}
            self.vis_info[m_name]['output'] = []
            self.vis_info[m_name]['grad'] = []

    def cat_info(self):
        ''' concatenate tensor list into one tensor'''
        for m_name in self.vis_layer_names:
            if isinstance(self.vis_info[m_name]['output'], list) and len(self.vis_info[m_name]['output']) > 0:
                self.vis_info[m_name]['output'] = torch.cat(
                    self.vis_info[m_name]['output'], dim=0)
            if isinstance(self.vis_info[m_name]['grad'], list) and len(self.vis_info[m_name]['grad']) > 0:
                self.vis_info[m_name]['grad'] = torch.cat(
                    self.vis_info[m_name]['grad'], dim=0)


def multiprocess_in_sequence(func, param_list, workers=10):
    param_data = [[] for _ in range(workers)]
    c_worker = 0
    for i, param in enumerate(param_list):
        param_data[c_worker].append((i, param))
        c_worker = (c_worker + 1) % workers

    q = multiprocessing.Queue()
    q.cancel_join_thread()
    count = 0

    def worker(func, param_part_list):
        for i, param in param_part_list:
            try:
                if isinstance(param, list):
                    data = func(*param)
                else:
                    data = func(param)
            except:
                continue
            q.put((i, data))

    for i in range(workers):
        w = multiprocessing.Process(
            target=worker,
            args=(func, param_data[i]))
        w.daemon = False
        w.start()

    data_list = [None for _ in range(len(param_list))]

    while count < len(param_list):
        i, data = q.get()
        data_list[i] = data
        count += 1

    return data_list
