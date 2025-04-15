import torch
import torch.nn.functional as F


class ConfModel(torch.nn.Module):
    def __init__(self, base_model, T, output_dim):
        super().__init__()
        self.model = base_model
        self.T = T.cuda()
        # self.T = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.T2 = torch.nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        if output_dim == 10:
            hidden_dim = 128
        else:
            hidden_dim = 256
        self.fc1 = torch.nn.Linear(output_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim_list[1], output_dim)

    # def forward(self, x, is_logit=False):
    #     with torch.no_grad():
    #         if is_logit:
    #             smx = F.softmax(x / self.T, dim = 1)
    #         else:
    #             output_raw = self.model(x)
    #             smx = F.softmax(output_raw / self.T, dim = 1)
        
    #     output_correction = smx
    #     # output_correction = F.dropout(output_correction, p=0.5, training=self.training)
    #     output_correction = self.fc1(output_correction).relu()
    #     output_correction = self.fc2(output_correction)
    #     # output_correction = self.fc3(output_correction)
    #     return output_correction


    def forward(self, x, is_logit=False):
        with torch.no_grad():
            if is_logit:
                output_raw = x
            else:
                output_raw = self.model(x)
            output_correction = F.softmax(output_raw / self.T, dim = 1)
        
        # output_correction = F.dropout(output_correction, p=0.5, training=self.training)
        output_correction = self.fc1(output_correction).relu()
        output_correction = self.fc2(output_correction)
        # output_correction = self.fc3(output_correction)
        return output_correction
    
    def get_T(self):
        return self.T.item()