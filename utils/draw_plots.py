import torch
from self_supervised_learning_sr.direct_optimize_polarization_SR_in_frequency_domain import SR_reconstruction
import numpy as np
import matplotlib.pyplot as plt
def compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader,save_file_directory, image_show=False):
    """
    :param SIM_data_dataloader:
    :param save_file_directory: The saving directory of plots
    :param image_show: The option to choose whether to display the plot results
    :return:
    """
    data_num = 0
    for SIM_data in SIM_data_dataloader:
        SIM_raw_data = SIM_data[0]
        SSIM_of_diff_input_num = torch.zeros(5, 1)
        PSNR_of_diff_input_num = torch.zeros(5, 1)
        for i in range(5):
            SSIM_of_diff_input_num[i, 0], PSNR_of_diff_input_num[i, 0], SR = SR_reconstruction(SIM_data,
                                                                                               input_num=i + 5,
                                                                                               image_show=image_show)
        if data_num == 0:
            SSIM = SSIM_of_diff_input_num
            PSNR = PSNR_of_diff_input_num
        else:
            SSIM = torch.cat([SSIM, SSIM_of_diff_input_num], 1)
            PSNR = torch.cat([PSNR, PSNR_of_diff_input_num], 1)
        data_num += 1
        if data_num > 30:
            break

    SSIM_mean = torch.mean(SSIM, 1).numpy()
    SSIM_std = torch.std(SSIM, 1).numpy()

    PSNR_mean = torch.mean(PSNR, 1).numpy()
    PSNR_std = torch.std(PSNR, 1).numpy()

    np.save(save_file_directory + "SSIM.npy", SSIM.numpy())
    np.save(save_file_directory + "PSNR.npy", PSNR.numpy())

    index = np.arange(5)
    total_width, n = 0.4, 2
    width = total_width / n

    plt.title('A Bar Chart')
    plt.bar(index, SSIM_mean, width=width, yerr=SSIM_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7,
            label='SSIM', color='#583d72')
    plt.legend(loc=2)
    plt.savefig(save_file_directory + 'SSIM_bar.eps', dpi=600, format='eps')
    plt.show()
    plt.bar(index - width, PSNR_mean, width=width, yerr=PSNR_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7,
            label='PSNR', color='#9f5f80')
    plt.xticks(index + 0.2, ['5', '6', '7', '8', '9'])
    plt.legend(loc=2)
    plt.grid(linestyle='--', c='#bbbbbb')
    plt.savefig(save_file_directory + 'PSNR_bar.eps', dpi=600, format='eps')
    plt.show()
