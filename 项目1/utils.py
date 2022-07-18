import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    #if cfg.save_fig:
    #    plt.savefig(cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()

def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')

