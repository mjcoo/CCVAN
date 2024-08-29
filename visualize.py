import matplotlib as plt
import pandas as pd
import seaborn as sns
import numpy as np
from rdkit import Chem
def condition_plot(zinc_path,mgcvae_path,ccvan_path):
    zinc = pd.read_csv(zinc_path)
    zinc = zinc[zinc['Length'] <= 16].reset_index(drop=True)
    labels = 16
    ticks = 18
    legends = 18
    figs = 3
    xl = [-8, 10]
    yl = [0, 1.4]
    alp = 0.2
    lw = 2
    lalp = 0.25
    clw = 5
    b = 10
    plt.figure(figsize=(figs*7, figs*4))
    kwargs = dict(hist_kws={'alpha':alp}, kde_kws={'linewidth':lw})
    n = 1
    for i in [0, 1]:
        for j in [2, 3]:
            plt.subplot(4, 5, n)
            ccvan = pd.read_csv(ccvan_path)
            mgcvae = pd.read_csv(mgcvae_path)
            plt.plot([i, i], [0, yl[1]], color="mediumaquamarine", alpha=lalp, linewidth=clw)
            plt.plot([j, j], [0, yl[1]], color="orange", alpha=lalp, linewidth=clw)
            #sns.distplot(zinc['logP'], color="orangered", label="ZINC (logP)", bins=b, **kwargs)
            #sns.distplot(zinc['MR']/10, color="lightslategrey", label="ZINC (MR/10)", bins=b, **kwargs)
            sns.distplot(ccvan['c1'], color="mediumaquamarine", label="ccvan (logP)", bins=b, **kwargs)
            sns.distplot(ccvan['c2']/10, color="orange", label="ccvan (MR/10)", bins=b, **kwargs)
            sns.distplot(mgcvae['C1'], color="blue", label="MGCVAE (logP)", bins=b, **kwargs)
            sns.distplot(mgcvae['C2']/10, color="pink", label="MGCVAE (MR/10)", bins=b, **kwargs)
            plt.xlim(xl)
            plt.xlabel(f'logP={i}, MR/10={j}', fontsize=labels)
            plt.xticks(fontsize=ticks)
            plt.ylim(yl)
            plt.ylabel('Density', fontsize=labels)
            plt.yticks(np.arange(yl[0], yl[1]+0.2, 0.2), fontsize=ticks)
            n += 1
            if n == 11:
                plt.legend(loc='upper left', fontsize=legends, bbox_to_anchor=(1.1, 0.9))
    plt.tight_layout(pad=0.5)
    plt.show()
    plt.savefig(r'C:\Users\admin\Desktop\物化条件生成.png')

def cvae_scores(CCVAN, cond_1, cond_2):
    cvae1 = CCVAN[(CCVAN['c1'] < cond_1+0.5) & (CCVAN['c1'] > cond_1-1.5)].reset_index(drop=True)
    cvae2 = CCVAN[(CCVAN['c2'] < cond_2*10+5) & (CCVAN['c2'] > cond_2*10-30)].reset_index(drop=True)
    cvae3 = CCVAN[(CCVAN['c2'] < cond_2*10+5) & (CCVAN['c2'] > cond_2*10-30)].reset_index(drop=True)
    return round(cvae1.shape[0]/CCVAN.shape[0]*100, 3), round(cvae2.shape[0]/CCVAN.shape[0]*100, 3), round(cvae3.shape[0]/CCVAN.shape[0]*100, 3), cond_1, cond_2*10
#cvae01 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_(0, 1).csv')
cvae02 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(0, 2).csv')
cvae03 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(0, 3).csv')
cvae04 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(0, 4).csv')
cvae05 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(0, 5).csv')
cvae06 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(0, 6).csv')
#cvae11 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_(1, 1).csv')
cvae12 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(1, 2).csv')
cvae13 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(1, 3).csv')
cvae14 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(1, 4).csv')
cvae15 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(1, 5).csv')
cvae16 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(1, 6).csv')
#cvae21 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_(2, 1).csv')
cvae22 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(2, 2).csv')
cvae23 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(2, 3).csv')
cvae24 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(2, 4).csv')
cvae25 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(2, 5).csv')
cvae26 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(2, 6).csv')
#cvae31 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_(3, 1).csv')
cvae32 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(3, 2).csv')
cvae33 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(3, 3).csv')
cvae34 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(3, 4).csv')
cvae35 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(3, 5).csv')
cvae36 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_lstm(3, 6).csv')
df = pd.DataFrame([#cvae_scores(cvae01, 0, 1),
                   cvae_scores(cvae02, 0, 2),
                   cvae_scores(cvae03, 0, 3),
                   cvae_scores(cvae04, 0, 4),
                   cvae_scores(cvae05, 0, 5),
                   cvae_scores(cvae06, 0, 6),
                   #cvae_scores(cvae11, 1,1),
                   cvae_scores(cvae12, 1, 2),
                   cvae_scores(cvae13, 1, 3),
                   cvae_scores(cvae14, 1, 4),
                   cvae_scores(cvae15, 1, 5),
                   cvae_scores(cvae16, 1, 6),
                   #cvae_scores(cvae21, 2, 1),
                   cvae_scores(cvae22, 2, 2),
                   cvae_scores(cvae23, 2, 3),
                   cvae_scores(cvae24, 2, 4),
                   cvae_scores(cvae25, 2, 5),
                   cvae_scores(cvae26, 2, 6),
                   #cvae_scores(cvae31, 3, 1),
                   cvae_scores(cvae32, 3, 2),
                   cvae_scores(cvae33, 3, 3),
                   cvae_scores(cvae34, 3, 4),
                   cvae_scores(cvae35, 3, 5),
                   cvae_scores(cvae36, 3, 6),
                   ],
                  columns=['Opt_logP (%)', 'Opt_MR (%)', 'Opt_Both (%)', 'C1_logP', 'C2_MR'])

print(df)

def cvae_scores(cvae, cond_1, cond_2):
    cvae1 = cvae[(cvae['C1'] < cond_1+0.5) & (cvae['C1'] > cond_1-1.5)].reset_index(drop=True)
    cvae2 = cvae[(cvae['C2'] < cond_2*10+5) & (cvae['C2'] > cond_2*10-30)].reset_index(drop=True)
    cvae3 = cvae1[(cvae1['C2'] < cond_2*10+5) & (cvae1['C2'] > cond_2*10-30)].reset_index(drop=True)
    return round(cvae1.shape[0]/cvae.shape[0]*100, 3), round(cvae2.shape[0]/cvae.shape[0]*100, 3), round(cvae3.shape[0]/cvae.shape[0]*100, 3), cond_1, cond_2*10
#cvae01 = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\zhibao_gen_(0, 1).csv')
cvae02 = pd.read_csv(r'D:\MGCVAE-main\results\generated_0_2.csv')
cvae12 = pd.read_csv(r'D:\MGCVAE-main\results\generated_1_2.csv')
cvae22 = pd.read_csv(r'D:\MGCVAE-main\results\generated_2_2.csv')
cvae32 = pd.read_csv(r'D:\MGCVAE-main\results\generated_3_2.csv')
cvae03 = pd.read_csv(r'D:\MGCVAE-main\results\generated_0_3.csv')
cvae13 = pd.read_csv(r'D:\MGCVAE-main\results\generated_1_3.csv')
cvae23 = pd.read_csv(r'D:\MGCVAE-main\results\generated_2_3.csv')
cvae33 = pd.read_csv(r'D:\MGCVAE-main\results\generated_3_3.csv')
cvae04 = pd.read_csv(r'D:\MGCVAE-main\results\generated_0_4.csv')
cvae14 = pd.read_csv(r'D:\MGCVAE-main\results\generated_1_4.csv')
cvae24 = pd.read_csv(r'D:\MGCVAE-main\results\generated_2_4.csv')
cvae34 = pd.read_csv(r'D:\MGCVAE-main\results\generated_3_4.csv')
cvae05 = pd.read_csv(r'D:\MGCVAE-main\results\generated_0_5.csv')
cvae15 = pd.read_csv(r'D:\MGCVAE-main\results\generated_1_5.csv')
cvae25 = pd.read_csv(r'D:\MGCVAE-main\results\generated_2_5.csv')
cvae35 = pd.read_csv(r'D:\MGCVAE-main\results\generated_3_5.csv')
cvae06 = pd.read_csv(r'D:\MGCVAE-main\results\generated_0_6.csv')
cvae16 = pd.read_csv(r'D:\MGCVAE-main\results\generated_1_6.csv')
cvae26 = pd.read_csv(r'D:\MGCVAE-main\results\generated_2_6.csv')
cvae36 = pd.read_csv(r'D:\MGCVAE-main\results\generated_3_6.csv')
df = pd.DataFrame([#cvae_scores(cvae01, 0, 1),
                   cvae_scores(cvae02, 0, 2),
                   cvae_scores(cvae03, 0, 3),
                   cvae_scores(cvae04, 0, 4),
                   cvae_scores(cvae05, 0, 5),
                   cvae_scores(cvae06, 0, 6),
                   #cvae_scores(cvae11, 1,1),
                   cvae_scores(cvae12, 1, 2),
                   cvae_scores(cvae13, 1, 3),
                   cvae_scores(cvae14, 1, 4),
                   cvae_scores(cvae15, 1, 5),
                   cvae_scores(cvae16, 1, 6),
                   #cvae_scores(cvae21, 2, 1),
                   cvae_scores(cvae22, 2, 2),
                   cvae_scores(cvae23, 2, 3),
                   cvae_scores(cvae24, 2, 4),
                   cvae_scores(cvae25, 2, 5),
                   cvae_scores(cvae26, 2, 6),
                   #cvae_scores(cvae31, 3, 1),
                   cvae_scores(cvae32, 3, 2),
                   cvae_scores(cvae33, 3, 3),
                   cvae_scores(cvae34, 3, 4),
                   cvae_scores(cvae35, 3, 5),
                   cvae_scores(cvae36, 3, 6),
                   ],
                  columns=['Opt_logP (%)', 'Opt_MR (%)', 'Opt_Both (%)', 'C1_logP', 'C2_MR'])

def sample_plot(file_path):
    mols=[]
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                mols.append(mol)
    img = Chem.Draw.MolsToGridImage(
        [m for m in mols if m is not None][:25], molsPerRow=5, subImgSize=(150, 150)
    )
    img.save(r'C:\Users\admin\Desktop\molecule.png')

def loss_plot():
    df = pd.read_csv(r'C:\Users\admin\Desktop\zhibao\loss.csv')
    d_losses =df['D Loss']
    g_losses=df['G Loss']
    real_scores=df['D(x)']
    fake_scores=df['D(G(z)']
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18,24))
    plot_data = [d_losses, g_losses, real_scores, fake_scores]
    legends = ["D Loss", "G Loss", "D(x)", "D(G(z))"]
    for i in range(4):
        axes[i][0].plot(plot_data[i], label=legends[i], alpha=0.8)
        #axes[i][0].grid()
        axes[i][0].legend()
        axes[i][1].plot(plot_data[i], label=legends[i], alpha=0.8)
        #axes[i][1].grid()
        axes[i][1].legend()
        # 设置刻度密度
        axes[i][0].set_yticks(np.linspace(axes[i][0].get_ylim()[0], axes[i][0].get_ylim()[1], num=1000))
        axes[i][1].set_yticks(np.linspace(axes[i][1].get_ylim()[0], axes[i][1].get_ylim()[1], num=1000))
        #axes[i][1].set_yscale('log')

    axes[3][0].set_xlabel("Steps")
    axes[3][1].set_xlabel("Steps")
    plt.show()


def calculate_novelty_by_smiles(generated_molecules,generated_smiles, reference_smiles):
    novelty_scores = 0
    validity = 0
    a=0
    for gen_smiles in generated_smiles :
        if gen_smiles in reference_smiles or pd.notna(gen_smiles):

            novelty_scores+=0
        else:
            novelty_scores+=1
    for generated_molecule in generated_molecules:
        if generated_molecule is None or Chem.MolToSmiles(generated_molecule) == '':
           a+=1
        else:
            validity += 1
    novelty_scores_all = float(novelty_scores/validity)
    validity = float(validity/5000)
    return validity,novelty_scores_all

