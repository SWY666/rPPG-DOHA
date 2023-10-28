# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import torch
from torch import nn
from einops import rearrange

class Conv_real_face_array(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(3, 64, kernel_size=(2, 8, 8), padding=(0, 2, 2))
        self.Conv_Layer2 = nn.Conv3d(64, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.BN1 = nn.BatchNorm3d(64)
        self.BN2 = nn.BatchNorm3d(32)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.Conv_Layer1(input)
        output = self.BN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)
        output = self.Conv_Layer2(output)
        output = self.BN2(output)
        output = self.Activation1(output)
        return output


class Conv_real_face_array2(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.Conv_Layer2 = nn.Conv3d(64, 8, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN1 = nn.BatchNorm3d(64)
        self.BN2 = nn.BatchNorm3d(8)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.Conv_Layer1(input)
        output = self.BN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)
        output = self.Conv_Layer2(output)
        output = self.BN2(output)
        output = self.Activation1(output)
        return output


class Conv_residual_array(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(3, 32, kernel_size=(1, 8, 8), padding=(0, 2, 2))
        self.Conv_Layer2 = nn.Conv3d(32, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.BN1 = nn.BatchNorm3d(32)
        self.BN2 = nn.BatchNorm3d(16)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)
        self.weight_init_all()

    def forward(self, input):
        output = self.Conv_Layer1(input)
        output = self.BN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)
        output = self.Conv_Layer2(output)
        output = self.Drop(output)
        output = self.BN2(output)
        output = self.Activation1(output)
        return output

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def weight_init_all(self):
        for child in self.children():
            self.weight_init(child)

class Conv_residual_array2(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN1 = nn.BatchNorm3d(32)
        self.Conv_Layer2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN2 = nn.BatchNorm3d(64)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)
        self.weight_init_all()

    def forward(self, input):
        output = self.Conv_Layer1(input)
        output = self.BN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)
        output = self.Conv_Layer2(output)
        output = self.Drop(output)
        output = self.BN2(output)
        output = self.Activation1(output)
        return output

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def weight_init_all(self):
        for child in self.children():
            self.weight_init(child)

class Conv_residual_array3(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(64, 32, kernel_size=(5, 3, 3), padding=(0, 0, 0))
        self.BN1 = nn.BatchNorm3d(32)
        self.Conv_Layer2 = nn.Conv3d(32, 8, kernel_size=(7, 3, 3), padding=(0, 0, 0))
        self.BN2 = nn.BatchNorm3d(8)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)
        self.weight_init_all()

    def forward(self, input):
        output = self.Conv_Layer1(input)
        output = self.BN1(output)
        output = self.Activation1(output)
        output = self.Conv_Layer2(output)
        output = self.Drop(output)
        output = self.BN2(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)
        return output

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def weight_init_all(self):
        for child in self.children():
            self.weight_init(child)

class Projection1(nn.Module):
    def __init__(self, args, num_of_multihead=4):
        super().__init__()
        # print("I'm provoked 8")
        dim = 8 * args.win_length
        self.heads = num_of_multihead
        self.num_of_multihead = self.heads
        self.to_q = nn.Linear(dim, dim)
        # self.to_q = nn.Sequential(nn.Linear(dim, dim*4), nn.Linear(dim*4, dim*4), nn.Linear(dim*4, dim))
        # self.to_q = nn.Identity()
        self.Drop = nn.Dropout(0.1)
        self.weight_init_all()

    def forward(self, input):
        projections = self.to_q(input)
        projections = self.Drop(projections)
        projection_space = projections.unsqueeze(1)
        attn_raw_heads = cal_cos_similarity_self(projection_space)
        attn_raw = attn_raw_heads.squeeze(1)
        return attn_raw

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def weight_init_all(self):
        for child in self.children():
            self.weight_init(child)


class Split_Module(nn.Module):
    def __init__(self, slice_lens=11):
        super().__init__()
        self.slice_lens = slice_lens

    def forward(self, input):
        input = rearrange(input, "b c t -> b t c")
        split_slices = []
        for i in range(input.shape[1] + 1 - self.slice_lens):
            split_slices.append(input[:, i:i + self.slice_lens, :].unsqueeze(1))
        split_result = torch.cat(split_slices, 1)
        split_result = rearrange(split_result, "b s c t -> b s (c t)")
        return split_result

def cal_cos_similarity_self(projection_space):
    projection_space = projection_space.squeeze(1)
    Numerator = projection_space @ projection_space.permute(0, 2, 1)
    Denominator = projection_space.mul(projection_space)
    Denominator = torch.sqrt(torch.sum(Denominator, dim=2, keepdim=True))
    Denominator = Denominator @ Denominator.permute(0, 2, 1)
    result = Numerator / Denominator
    return result

class super_fusion(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.c = c
        self.Linear_layer = nn.Linear(self.c, 1)
        self.Activation = nn.Sigmoid()

    def forward(self, motion_mask, appearance_mask):
        coefficient = appearance_mask.shape[-1] * appearance_mask.shape[-2]
        appearance_compression = self.Linear_layer(rearrange(appearance_mask, "b c t h w -> b t h w c")).squeeze(-1)
        appearance_compression = self.Activation(appearance_compression)
        appearance_compression = rearrange(appearance_compression, "b t h w -> (b t) h w")
        appearance_compression_slice = torch.split(appearance_compression, 1, 0)
        appearance_compression_final = []
        for index in range(len(appearance_compression_slice)):
            motion_image_weight = coefficient * appearance_compression_slice[index] / (2 * torch.norm(appearance_compression_slice[index], 1))
            appearance_compression_final.append(motion_image_weight)
        motion_image_weight = torch.cat(appearance_compression_final, 0)
        motion_image_weight = rearrange(motion_image_weight, "(b t) h w -> b t h w", b=motion_mask.shape[0])
        motion_mask = motion_mask.mul(motion_image_weight.unsqueeze(1))
        return motion_mask, motion_image_weight

class Ultimate_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Conv_real_face_array = Conv_real_face_array(args.length)  # step1
        self.Conv_real_face_array2 = Conv_real_face_array2(args.length)  # step2
        self.Conv_residual_array = Conv_residual_array(args.length)
        self.Conv_residual_array2 = Conv_residual_array2(args.length)
        self.Conv_residual_array3 = Conv_residual_array3(args.length)
        self.super_fusion = super_fusion()
        self.super_fusion2 = super_fusion(8)
        self.split_module = Split_Module(args.win_length)
        self.GLOBAL_AVG = nn.AvgPool3d((1, 14, 14))
        self.Projection1 = Projection1(args)

    def forward(self, input_residual, input_real_face_array):
        output = self.Conv_real_face_array(input_real_face_array)
        output_R = self.Conv_residual_array(input_residual)
        output_R, face_mask = self.super_fusion(output_R, output)

        output = self.Conv_real_face_array2(output)
        output_R = self.Conv_residual_array2(output_R)
        output_R1, face_mask2 = self.super_fusion2(output_R, output)

        output_R1 = self.Conv_residual_array3(output_R1)
        output_R = self.GLOBAL_AVG(output_R1).squeeze(-1).squeeze(-1)

        output_R = self.split_module(output_R)
        attn = self.Projection1(output_R)
        return attn, face_mask, face_mask2


class Ultimate_model_wave(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Conv_real_face_array = Conv_real_face_array(args.length)  # step1
        self.Conv_real_face_array2 = Conv_real_face_array2(args.length)  # step2
        self.Conv_residual_array = Conv_residual_array(args.length)
        self.Conv_residual_array2 = Conv_residual_array2(args.length)
        self.Conv_residual_array3 = Conv_residual_array3(args.length)
        self.super_fusion = super_fusion()
        self.super_fusion2 = super_fusion(8)
        self.split_module = Split_Module(args.win_length)
        self.GLOBAL_AVG = nn.AvgPool3d((1, 14, 14))
        self.Projection1 = Projection1(args)

    def forward(self, input_residual, input_real_face_array):
        output = self.Conv_real_face_array(input_real_face_array)
        output_R = self.Conv_residual_array(input_residual)
        output_R, face_mask = self.super_fusion(output_R, output)

        output = self.Conv_real_face_array2(output)
        output_R = self.Conv_residual_array2(output_R)
        output_R1, face_mask2 = self.super_fusion2(output_R, output)

        output_R1 = self.Conv_residual_array3(output_R1)
        output_R = self.GLOBAL_AVG(output_R1).squeeze(-1).squeeze(-1)
        output_R = torch.mean(output_R, dim=1)
        return output_R, face_mask, face_mask2




class Ultimate_model_HR_version(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Conv_real_face_array = Conv_real_face_array(args.length)  # step1
        self.Conv_real_face_array2 = Conv_real_face_array2(args.length)  # step2
        self.Conv_residual_array = Conv_residual_array(args.length)
        self.Conv_residual_array2 = Conv_residual_array2(args.length)
        self.Conv_residual_array3 = Conv_residual_array3(args.length)
        self.super_fusion = super_fusion()
        self.super_fusion2 = super_fusion(8)
        self.split_module = Split_Module(args.win_length)
        self.GLOBAL_AVG = nn.AvgPool3d((1, 14, 14))
        self.Projection1 = Projection1(args)

    def forward(self, input_residual, input_real_face_array):
        output = self.Conv_real_face_array(input_real_face_array)
        output_R = self.Conv_residual_array(input_residual)
        output_R, face_mask = self.super_fusion(output_R, output)

        output = self.Conv_real_face_array2(output)
        output_R = self.Conv_residual_array2(output_R)
        output_R1, face_mask2 = self.super_fusion2(output_R, output)

        output_R1 = self.Conv_residual_array3(output_R1)
        output_R = self.GLOBAL_AVG(output_R1).squeeze(-1).squeeze(-1)

        output_R = self.split_module(output_R)
        attn = self.Projection1(output_R)
        return attn, face_mask, face_mask2


def up_down(tensors):
    tensor_result = torch.zeros(tensors.shape)
    for i in range(tensors.shape[0]):
        tensor_result[i, :] = tensors[tensors.shape[0]-i-1, :]
    return tensor_result

class Transformer1(nn.Module):
    def __init__(self, num_of_multihead=4, dim=88, dim_head=22):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attend2 = nn.Softmax(dim=-2)
        self.heads = num_of_multihead
        inner_dim = dim_head * self.heads
        self.num_of_multihead = self.heads
        self.L1 = nn.Linear(inner_dim, dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.FCN = nn.Linear(dim_head, 1)
        self.to_out = nn.Sequential(
            self.L1,
            nn.Dropout(0.1),
        )

    def forward(self, input):
        q = rearrange(self.to_q(input), 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(self.to_k(input), 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(self.to_v(input), 'b t (h d) -> b h t d', h=self.heads)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = torch.cat(torch.split(out, 1, 1), dim=-1).squeeze(1)
        out = self.to_out(out)
        return out

class Transformer2(nn.Module):
    def __init__(self, num_of_multihead=4, dim=88, dim_head=22):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attend2 = nn.Softmax(dim=-2)
        self.heads = num_of_multihead
        inner_dim = dim_head * self.heads
        self.num_of_multihead = self.heads
        self.L1 = nn.Linear(inner_dim, dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.FCN = nn.Linear(dim_head, 1)
        self.to_out = nn.Sequential(
            self.L1,
            nn.Dropout(0.1),
        )

    def forward(self, input):
        q = rearrange(self.to_q(input), 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(self.to_k(input), 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(self.to_v(input), 'b t (h d) -> b h t d', h=self.heads)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = torch.cat(torch.split(out, 1, 1), dim=-1).squeeze(1)
        out = self.to_out(out)
        return out

if __name__ == "__main__":
    layer1 = Conv_residual_array()
    for child in layer1.children():
        print(isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv3d)))

