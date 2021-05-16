from types import FunctionType
import torch
import torch.nn.functional as F


class GradCam(object):
    def __init__(self, model, vis_layer_names, detach=True):
        self.model = model
        self.vis_layer_names = vis_layer_names
        self.init_vis_layers(detach)

    def _add_hook(self, net, detach):
        if hasattr(net, '_modules'):
            for m_name, module in net._modules.items():
                if m_name in self.vis_layer_names:
                    print(m_name)
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
                    h = module.register_full_backward_hook(save_gradient)
                    # h = module.register_full_backward_hook(self.save_gradient)
                    self.backward_hook_handles.append(h)
                self._add_hook(module, detach)

    def add_hook(self, detach):
        self._add_hook(self.model, detach)

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

    def cal_grad(self, y, t_label):
        """
        Args:
            model:
            imgs: NHWC
            t_label: target label to be visualized
        """
        one_hots = torch.zeros(1, y.shape[-1]).to(y.device)
        one_hots[0, t_label] = 1
        y.backward(gradient=one_hots, retain_graph=True)

    def cal_cam(self, ind):
        result = dict()
        for key in self.vis_info.keys():
            grads_val = self.vis_info[key]['grad'][0][ind] # (C, H, W)
            feature = self.vis_info[key]['output'][ind]
            weights = torch.mean(grads_val, dim=(1, 2), keepdim=True)
            cam = weights * feature
            cam = torch.sum(cam, dim=0)
            cam = F.relu(cam) # (1, H, W)
            # normalize to (0, 1)
            max_value = torch.max(cam) # (1)
            min_value = torch.min(cam) # (1)
            cam.sub_(min_value).div_(max_value-min_value)
            # save to vis_info
            result[key] = cam[0].cpu().detach()
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
                self.vis_info[m_name]['output'] = torch.cat(self.vis_info[m_name]['output'], dim=0)
            if isinstance(self.vis_info[m_name]['grad'], list) and len(self.vis_info[m_name]['grad']) > 0:
                self.vis_info[m_name]['grad'] = torch.cat(self.vis_info[m_name]['grad'], dim=0)
