import torch


def wireless_channel(x, mode="ideal", power=None, SNRdB=None, **kwargs):
    """
    功率归一化和信道选择的综合函数。

    参数：
        x: torch.Tensor，输入信号。
        mode: str，信道类型：
            - "ideal": 无失真信道
            - "awgn": 加性高斯白噪声信道
            - "rayleigh": 瑞利衰落信道
            - "nakagami": Nakagami-m信道
            - "rician": 莱斯信道
        power: float，目标功率，用于功率归一化。
        SNRdB: float，信噪比，用于加性高斯白噪声、瑞利衰落、Nakagami-m和莱斯信道。
        m: float，Nakagami-m信道的m参数，仅在mode="nakagami"时需要。
        K: float，莱斯信道的K因子，仅在mode="rician"时需要。

    返回：
        torch.Tensor，处理后的信号。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k = kwargs.get('k', None)
    m = kwargs.get('m', None)
    # 1. 功率归一化
    x_power = torch.flatten(x)  # 展平输入
    power = torch.tensor(power)
    Double_k = x_power.shape[0]

    # 检查长度是否为偶数
    if Double_k % 2 != 0:
        raise ValueError("Input length must be even to form a complex signal.")

    # 构造复数信号
    z_tilde = x_power[:Double_k // 2] + x_power[Double_k // 2:] * 1j
    single_k = z_tilde.shape[0]

    # 计算归一化因子
    norm_factor = torch.sqrt((z_tilde.conj().T @ z_tilde))
    z = torch.sqrt(power * single_k) * z_tilde / norm_factor

    # 2. 信道处理
    if mode == "ideal":
        # 无失真信道
        return z

    elif mode == "awgn":
        if SNRdB is None:
            raise ValueError("SNRdB must be provided for AWGN channel.")

        with torch.no_grad():
            signal_power = torch.mean(z.real ** 2) + torch.mean(z.imag ** 2)
            # 将SNR从dB转换为线性
            SNRdB_linear = 10 ** (SNRdB / 10)

            # 计算噪声标准差
            noise_power = signal_power / SNRdB_linear
            noise_std = torch.sqrt(noise_power / 2.0)  # 噪声被分配到实部和虚部

            # 生成噪声
            noise_real = torch.normal(mean=0.0, std=noise_std.item(), size=z.real.shape)
            noise_imag = torch.normal(mean=0.0, std=noise_std.item(), size=z.imag.shape)

            # 复数噪声
            noise = noise_real + 1j * noise_imag

        z_hat = z + noise

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 3. 将 z_hat 转换为与输入 x 相同的维度
    z_real = z_hat.real
    z_imag = z_hat.imag
    z_flattened = torch.cat((z_real, z_imag), dim=-1)  # 拼接实部和虚部

    return z_flattened.view_as(x)  # 调整维度与 x 一致

# x = torch.randn(1, 3, 32, 32)

# out = wireless_channel(x,"awgn",1,5)