from rdkit import RDLogger
import torch
import pandas as pd
RDLogger.DisableLog("rdApp.*")
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv(r'D:\data\proteins_data\cyp2d6\datasets\ZINC25W.csv')

charset = sorted(list(set(''.join(df['smiles'].values))))
charset_length = len(charset)
max_length = max(map(len, df['smiles']))

def string_to_one_hot(smiles_string):
    one_hot = np.zeros((max_length, charset_length), dtype=np.int8)
    for i, c in enumerate(smiles_string):
        one_hot[i, charset.index(c)] = 1
    return one_hot

def one_hot_to_string(one_hot):
    return ''.join(charset[i] for i in one_hot.argmax(axis=1))


one_hot_list = list(map(string_to_one_hot, df['smiles']))
smiles_tensor = torch.tensor(one_hot_list, dtype=torch.float32)

vocab_freq =  {}
word_length_dist = []
for smile in df['smiles']:
    for s in smile:
        if s not in vocab_freq.keys():
            vocab_freq[s] = 0
        vocab_freq[s] += 1
    word_length_dist.append(len(smile))
vocab = list(vocab_freq.keys())


N_HIDDEN_gan = 1024
N_HIDDEN_gan1 = 512
N_HIDDEN_gan2 = 256
SMILES_MAXLEN = max_length
NUM_EPOCHS = 500
BATCH_SIZE = 420
N_INPUT = 3640
N_HIDDEN_vae = 400
N_Z = 20
G_LEARNING_RATE = 5e-4
D_LEARNING_RATE = 5e-4
latent_dim = 50
input_size = 35

def smile2vec(vocab, vecsize, smile):
    vec = []
    for i in range(vecsize):
        v = [0 for _ in range(len(vocab))]
        if i < len(smile):
            v[vocab.index(smile[i])] = 1
        vec += v
    return vec

X = []

for smile in list(df['smiles']):
    X.append(smile2vec(vocab, SMILES_MAXLEN, smile))

X = np.array(X)
X_train_tensor = torch.tensor(X)

dataset = TensorDataset(X_train_tensor)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def property_weight(out_tensor,logp):
    true=[]
    best_acc = 0.0
    for vec,logp_value in zip(out_tensor,logp):
        vec = vec.reshape(SMILES_MAXLEN, len(vocab))
        smile = "".join([vocab[torch.argmax(v).item()] for v in vec])
        mol = Chem.MolFromSmiles(smile)
        while not mol:
            if len(smile) == 0: break
            smile = smile[:-1]
            mol = Chem.MolFromSmiles(smile)
        mol = Chem.MolFromSmiles(smile)
        logp_pre = Descriptors.MolWt(mol)
        logp_pre = round(logp_pre)
        logp_value= round(logp_value.item())
        if logp_pre == logp_value:
            true.append(smile)
            print(len(true))
    weight = (len(out_tensor)/len(true))

    return weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == in_dim, "in_dim must be divisible by num_heads"

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, N, C = x.size()
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = F.softmax(torch.bmm(Q.reshape(-1, N, self.head_dim), K.reshape(-1, self.head_dim, N)), dim=-1)
        out = torch.bmm(attn_weights, V.reshape(-1, N, self.head_dim))
        out = out.reshape(B, self.num_heads, N, self.head_dim).transpose(1, 2).contiguous().reshape(B, N, -1)
        out = self.out(out)
        return out

class CVAE(nn.Module):
    def __init__(self, n_input=N_INPUT,embedding_dim=128, n_hidden1=512, n_hidden2=256, n_z=100):
        super(CVAE, self).__init__()
        #self.attention = MultiHeadSelfAttention(512, 4)
        self.lstm_encode = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden1)
        self.lstm_decode = nn.LSTM(input_size=input_size, hidden_size=n_hidden1, batch_first=True)
        self.embedding_layer = nn.Linear(input_size, embedding_dim)
        self.fc1 = nn.Linear(53248, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_z)
        self.fc3 = nn.Linear(n_hidden1, n_z)
        self.fc4 = nn.Linear(n_z, n_hidden1)
        self.fc5 = nn.Linear(n_hidden2, n_hidden1)
        self.fc6 = nn.Linear(n_hidden1, n_input)

        self.attention = nn.Linear(n_hidden1, 1)
        self.attn = nn.Linear(n_hidden1 * 2, max_length)

    def encode(self, x):
        x = x.view(-1, max_length, input_size)
        x = self.embedding_layer(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        x_out, (ht, ct) = self.lstm_encode(x)
        att_input = torch.cat([x_out, ht[-1].unsqueeze(0).repeat(x_out.size(0), 1, 1)], dim=2)
        attn_weights = torch.softmax(self.attn(att_input),dim=2)
        attn_weights = attn_weights.permute(1,0,2)
        x_out = x_out.permute(1,0,2)
        attn_applied = torch.bmm(attn_weights, x_out)
        h = attn_applied.mean(1)
        h = torch.relu(self.fc1(h))

        return self.fc2(h), self.fc3(h)


    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu((self.fc4(z)))
        #h = h.view(-1,max_length,input_size)

        #ht,hidden = self.lstm_decode(h)
        #ht = ht.mean(1)

        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class GAN(torch.nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.attention = MultiHeadSelfAttention(35, 5)
        self.model = (
        torch.nn.Sequential(
        torch.nn.Linear(N_INPUT, N_HIDDEN_gan),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(N_HIDDEN_gan, N_HIDDEN_gan1),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(N_HIDDEN_gan1, N_HIDDEN_gan2),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(N_HIDDEN_gan2, 1),
        torch.nn.Sigmoid()))

    def forward(self, input):
        input = input.view(-1,max_length,input_size)
        input = self.attention(input)
        input = input.view(input.size(0),-1)
        out = self.model(input)
        return out

def get_best_smile(out_tensor):
    smi_list=[]
    for vec in out_tensor:
        vec = vec.reshape(SMILES_MAXLEN, len(vocab))
        smile = "".join([vocab[torch.argmax(v).item()] for v in vec])
        mol = Chem.MolFromSmiles(smile)
        while not mol:
            if len(smile) == 0: break
            smile = smile[:-1]
            mol = Chem.MolFromSmiles(smile)
        mol = Chem.MolFromSmiles(smile)
        num_atoms = mol.GetNumAtoms()
        smi_list.append(smile)

    return smi_list

def generate_smiles(cvae,batch) :
    cvae.eval()
    z = torch.randn(batch, latent_dim*2)
    with torch.no_grad():
        output = cvae.decode(z, c1)
        smiles = get_best_smile(output)

    return smiles

def draw_molecules(cvae) :
    generated_smiles = generate_smiles(cvae,batch=5000)
    cvae_df = pd.DataFrame({'SMILES': generated_smiles})
    cvae_df.to_csv(f'C:/Users/admin/Desktop/zhibao/zhibao_gen_vina.csv', index=False)
    print(f'Saving zhibao_gen_.csv ({cvae_df.shape[0]})...')


gan = GAN()
#vae = VAE(n_input=N_INPUT, n_hidden=N_HIDDEN_vae, n_z=N_Z)
cvae = CVAE()
g_optimizer = torch.optim.Adam(cvae.parameters(), lr=G_LEARNING_RATE)
d_optimizer = torch.optim.Adam(gan.parameters(), lr=D_LEARNING_RATE)

losses = []
reconst_losses = []
kl_divs = []
d_losses = []
g_losses = []
real_scores = []
fake_scores = []
criterion = torch.nn.BCELoss()
total_step = len(data_loader)
smi = []
SAscore = []
smi_list = []
for epoch in range(NUM_EPOCHS):
    novel_smi = []
    valid_smi = []
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for i, (x, c1) in enumerate(data_loader):
        #cond2_batch = cond2_tensor[i:i+BATCH_SIZE]
        cond1_batch = c1.unsqueeze(1)
        #cond1_batch = c1
        #cond2_batch = c2
        x_batch = x.float()
        x_batch = x_batch.view(x_batch.size(0), -1)
        x_reconst, mu, log_var = cvae(x_batch,cond1_batch)
        outputs_real = gan(x_batch)
        real_labels = torch.ones(outputs_real.shape[0], 1)
        d_loss_real = criterion(outputs_real, real_labels)
        outputs_fake = gan(x_reconst)
        real_score = outputs_real
        fake_score = outputs_fake
        fake_labels = torch.zeros(outputs_fake.shape[0], 1)
        d_loss_fake = criterion(outputs_fake, real_labels)
        d_loss = torch.mean(outputs_fake) - torch.mean(outputs_real)
        #d_loss = d_loss_real+d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        if (i%4) == 0:
            d_loss.backward()
        d_optimizer.step()
        x_reconst, mu, log_var = cvae(x_batch,cond1_batch)
        fake_outputs = gan(x_reconst)
        real_labels = torch.ones(fake_outputs.shape[0], 1)
        #g_loss = criterion(fake_outputs, real_labels)
        g_loss = - torch.mean(fake_outputs)
        reconst_loss = torch.nn.functional.binary_cross_entropy(x_reconst, x_batch, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        g_loss_ = reconst_loss + kl_div + g_loss
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss_.backward()
        g_optimizer.step()
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        real_scores.append(real_score.mean().item())
        fake_scores.append(fake_score.mean().item())
        print('Epoch [{}/{}], Step [{}/{}], d_loss: {:}, g_loss: {:}'
            .format(epoch+1, NUM_EPOCHS, i + 1, total_step, d_loss.item(), g_loss.item()))

    with torch.no_grad():
        z = torch.randn(BATCH_SIZE, latent_dim * 2)
        out = cvae.decode(z)
        for vec in out:
            vec = vec.reshape(SMILES_MAXLEN, len(vocab))
            smile = "".join([vocab[torch.argmax(v).item()] for v in vec])
            mol = Chem.MolFromSmiles(smile)
            while not mol:
                if len(smile) == 0: break
                smile = smile[:-1]
                mol = Chem.MolFromSmiles(smile)
            mol = Chem.MolFromSmiles(smile)
            num_atoms = mol.GetNumAtoms()
            valid_smi.append(smile)
            if smile not in df['smiles']:
                novel_smi.append(smile)
    cvae_df_ = pd.DataFrame({'SMILES': novel_smi})
    cvae_df =  pd.DataFrame({'SMILES': valid_smi})
    cvae_df_.to_csv(f'C:/Users/admin/Desktop/zhibao/zhibao_gen_cvae_novel.csv', index=False)
    print(f'novel({cvae_df_.shape[0]})...')
    cvae_df.to_csv(f'C:/Users/admin/Desktop/zhibao/zhibao_gen_cvae_valid.csv', index=False)
    print(f'valid ({cvae_df.shape[0]})...')

    print(smi_list)
    print("Epoch[{}/{}]".format(epoch+1, NUM_EPOCHS))
    #best_smiles = get_best_smile(x_reconst)
    #best_mol =  Chem.MolFromSmiles(best_smiles)
    #smi.append(Chem.MolToSmiles(best_mol))
    #SAscore.append(sascorer.calculateScore(best_mol))
best_smiles = draw_molecules(cvae)
test_df = pd.DataFrame(smi_list)
test_df.to_csv(r'C:\Users\admin\Desktop\AMPC\test.csv')
loss = pd.DataFrame({'D Loss':d_losses ,'G Loss':g_losses})
loss.to_csv(f'C:/Users/admin/Desktop/zhibao/loss_WGAN.csv', index=False)


