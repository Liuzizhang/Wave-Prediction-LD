# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, \
                            mean_absolute_percentage_error, \
                            mean_squared_error,  \
                            r2_score
import signal
import subprocess
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchinfo import summary
import torch.nn.functional as F
np.random.seed(0)
torch.manual_seed(0)
def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  
    rmse = np.sqrt(mse)  
    return rmse

# %%
# The dataset is not included in this code. You need to download and organize the dataset according to the instructions in the README.md file.
data = pd.read_csv('waves-test.csv')
data.tail(5).style.background_gradient()

print(type(data['Hs'].iloc[0]),type(data['Date/Time'].iloc[0]))
# Let's convert the data type of timestamp column to datatime format
data['Date/Time'] = pd.to_datetime(data['Date/Time'], format="%Y/%m/%d %H:%M")
print(type(data['Hs'].iloc[0]),type(data['Date/Time'].iloc[0]))

print(data.isnull().sum())
data.dropna(inplace=True)
# Selecting subset

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Date/Time'], data['Hs'], color='darkorange' ,label='Waves')

locator = mdates.AutoDateLocator(minticks=8, maxticks=12)  
formatter = mdates.DateFormatter('%Y-%m-%d %H:%M') 
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.title('Waves high', 
          fontdict={'family': 'Times New Roman', 'fontsize': 16, 'color':'green'})
plt.xticks(rotation=45) 
plt.ylabel('Hs', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.legend(loc="upper right", prop={'family': 'Times New Roman'})
plt.show()


# %% [markdown]


# %%
features = data.drop(['Hs', 'Date/Time'], axis=1)
target = data['Hs'].values.reshape(-1, 1)
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target)

print(features_scaled.shape, target_scaled.shape)
time_steps = 144
X_list = []
y_list = []

for i in range(len(features_scaled) - time_steps):
    X_list.append(features_scaled[i:i+time_steps])
    y_list.append(target_scaled[i+time_steps])
X = np.array(X_list)
y = np.array(y_list)
samples, time_steps, num_features = X.shape  
train_ratio, val_ratio = 0.7, 0.15
train_size = int(len(X) * train_ratio)
val_size = int(len(X) * val_ratio)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

X_train = scaler_features.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler_features.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test = scaler_features.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

y_train = scaler_target.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val = scaler_target.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test = scaler_target.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SparseMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, sparsity=0.5):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity = sparsity

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: (B, T, E)
            key: (B, S, E)
            value: (B, S, E)
        Returns:
            attn_output: (B, T, E)
            attn_weights: (B, H, T, S)
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)  # T
        src_len = key.size(1)   # S

        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)     # (B, H, S, D)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, S, D)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, S)

        if self.sparsity < 1.0:
            num_keep = int(src_len * self.sparsity)
            
            topk_values, topk_indices = torch.topk(attn_scores, k=num_keep, dim=-1)  # (B, H, T, K)
            
            sparse_mask = torch.zeros_like(attn_scores).scatter_(
                dim=-1, 
                index=topk_indices, 
                src=torch.ones_like(topk_values))
            
            attn_scores = attn_scores.masked_fill(~sparse_mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, T, S)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim)  # (B, T, E)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class LSTM_Transformer_SparseAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_heads, transformer_layers, output_dim, sparsity=0.5, dropout=0.5):
        super(LSTM_Transformer_SparseAttention, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        
        self.sparse_attention = SparseMultiheadAttention(embed_dim=hidden_dim, num_heads=transformer_heads, dropout=dropout, sparsity=sparsity)
        
    
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=transformer_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=transformer_layers)
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        assert len(x.shape) == 3, f"Input should be 3D (batch_size, seq_len, input_dim), got {x.shape}"
        assert x.shape[-1] == self.lstm.input_size, f"Last dimension of input should match input_dim ({self.lstm.input_size}), got {x.shape[-1]}"
        
        lstm_out, _ = self.lstm(x)
        
        sparse_attn_out, _ = self.sparse_attention(lstm_out, lstm_out, lstm_out)
        
        transformer_out = self.transformer_encoder(sparse_attn_out)
        
        output = self.fc(transformer_out[:, -1, :])
        return output

input_dim = num_features  
hidden_dim = 96 
lstm_layers = 2
transformer_heads = 4  
transformer_layers = 2 
output_dim = 1  
sparsity = 0.5  
dropout = 0.5  
learning_rate = 1e-4  
weight_decay = 1e-2 

model = LSTM_Transformer_SparseAttention(
    input_dim=input_dim, 
    hidden_dim=hidden_dim, 
    lstm_layers=lstm_layers, 
    transformer_heads=transformer_heads, 
    transformer_layers=transformer_layers, 
    output_dim=output_dim, 
    sparsity=sparsity, 
    dropout=dropout
)

criterion_mse = nn.MSELoss()  
criterion_mae = nn.L1Loss()  
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


batch_size = 32  
num_features = input_dim 
summary(model, (batch_size, time_steps, num_features))




num_epochs = 200  
CHECKPOINT_PATH = 'training_checkpoint.pth'
GPU_ID = 0
MAX_RESTARTS = 3

def train(model, iterator, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    epoch_loss_mse = 0
    epoch_loss_mae = 0

    model.train()  
    for batch in iterator:
        optimizer.zero_grad()  
        inputs, targets = batch  
        inputs, targets = inputs.to(device), targets.to(device)  
        outputs = model(inputs)  

        loss_mse = criterion_mse(outputs, targets)  
        loss_mae = criterion_mae(outputs, targets)

        combined_loss = loss_mse 

        combined_loss.backward()
        optimizer.step()

        epoch_loss_mse += loss_mse.item()  
        epoch_loss_mae += loss_mae.item()

    average_loss_mse = epoch_loss_mse / len(iterator) 
    average_loss_mae = epoch_loss_mae / len(iterator)
    return average_loss_mse, average_loss_mae

def evaluate(model, iterator):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    epoch_loss_mse = 0
    epoch_loss_mae = 0

    model.eval()  
    with torch.no_grad():  
        for batch in iterator:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  
            outputs = model(inputs)  

            loss_mse = criterion_mse(outputs, targets)  
            loss_mae = criterion_mae(outputs, targets)

            epoch_loss_mse += loss_mse.item()  
            epoch_loss_mae += loss_mae.item()

    return epoch_loss_mse / len(iterator), epoch_loss_mae / len(iterator)


def kill_zombie_processes():
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()[0].decode('utf-8')
        

        zombie_pids = []
        for line in out_str.split('\n'):
            if line.strip() and 'python' in line.lower():
                pid, name, mem = line.split(', ')
                if int(mem) > 0:  
                    zombie_pids.append(int(pid))
  
        for pid in zombie_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"killed PID: {pid}")
            except ProcessLookupError:
                continue
                
        time.sleep(5)  
        
    except Exception as e:
        print(f"error: {str(e)}")

def gpu_health_check():
    try:
        torch.cuda.empty_cache()
        dummy_tensor = torch.randn(1000, 1000, device=f'cuda:{GPU_ID}')
        del dummy_tensor
        return True
    except RuntimeError as e:
        print(f"GPU error: {str(e)}")
        return False


epoch = 200
train_mselosses = []
valid_mselosses = []
test_mselosses =[]
train_maelosses = []
valid_maelosses = []
test_maelosses =[]


restart_count = 0
while restart_count < MAX_RESTARTS:
    try:
        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            print("check point...")
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_mselosses = checkpoint['train_mselosses']
            valid_mselosses = checkpoint['valid_mselosses']
            test_mselosses = checkpoint['valid_mselosses']
            train_maelosses = checkpoint['train_maelosses']
            valid_maelosses = checkpoint['valid_maelosses']
            test_maelosses = checkpoint['valid_maelosses']
            print(f"epoch {start_epoch},continue train...")

        if not gpu_health_check():
            print("clear..")
            kill_zombie_processes()
            torch.cuda.empty_cache()
            if not gpu_health_check():
                print("GPU not enough")
                break

        for current_epoch in range(start_epoch, num_epochs):
            try:
                train_loss_mse, train_loss_mae = train(model, train_loader, optimizer)
                valid_loss_mse, valid_loss_mae = evaluate(model, val_loader)
                test_loss_mse, test_loss_mae = evaluate(model, test_loader)

                train_mselosses.append(train_loss_mse)
                valid_mselosses.append(valid_loss_mse)
                test_mselosses.append(test_loss_mse)
                train_maelosses.append(train_loss_mae)
                valid_maelosses.append(valid_loss_mae)
                test_maelosses.append(test_loss_mae)

                print(f'Epoch: {current_epoch+1:02}, Train MSELoss: {train_loss_mse:.5f}, Train MAELoss: {train_loss_mae:.3f}, Val. MSELoss: {valid_loss_mse:.5f}, Val. MAELoss: {valid_loss_mae:.3f},Test. MSELoss: {test_loss_mse:.5f}, Test. MAELoss: {test_loss_mae:.3f}')

                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_mselosses': train_mselosses,
                    'valid_mselosses': valid_mselosses,
                    'test_mselosses': test_mselosses,
                    'train_maelosses': train_maelosses,
                    'valid_maelosses': valid_maelosses,
                    'test_maelosses': test_maelosses,
                }, CHECKPOINT_PATH)

            except Exception as e:
                print(f"training error epoch {current_epoch+1},error:{str(e)}")
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'emergency_checkpoint.pth')
                raise

        break

    except Exception as e:
        print(f"training error: {str(e)}")
        kill_zombie_processes()
        torch.cuda.empty_cache()
        restart_count += 1
        print(f" {restart_count}/{MAX_RESTARTS} remake...")
        time.sleep(10)

if restart_count >= MAX_RESTARTS:
    print("get max remake,training stop")
else:
    print("training finish")
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("training finish clear CHECKPOINT")


folder_name = "result"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_mselosses, label='Train MSELoss')
plt.plot(valid_mselosses, label='Validation MSELoss')
plt.plot(test_mselosses, label='Test MSELoss')
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.title('Train,Validation and Test MSELoss')
plt.legend()
plt.grid(True)
plt.savefig('result/mse_loss_plot.png')


plt.subplot(1, 2, 2)
plt.plot(train_maelosses, label='Train MAELoss')
plt.plot(valid_maelosses, label='Validation MAELoss')
plt.plot(test_maelosses, label='Test MAELoss')
plt.xlabel('Epoch')
plt.ylabel('MAELoss')
plt.title('Train,Validation and Test MAELoss')
plt.legend()
plt.grid(True)
plt.savefig('result/mae_loss_plot.png')
plt.show()



def prediction(model, iterator):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in iterator:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    return all_targets, all_predictions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
targets, predictions = prediction(model, val_loader)

denormalized_targets = scaler_target.inverse_transform(targets)
denormalized_predictions = scaler_target.inverse_transform(predictions)

df = pd.DataFrame(denormalized_predictions, columns=['predictions'])
df['targets'] = denormalized_targets  
df.to_csv('result/predictions_LSTM_Transformer.csv', index=False)  

# Visualize the data
plt.figure(figsize=(12,6))
plt.style.use('_mpl-gallery')
plt.title('Comparison of validation set prediction results')
plt.plot(denormalized_targets, color='blue',label='Actual Value')
plt.plot(denormalized_predictions, color='orange', label='Valid Value')
plt.legend()
plt.show()
plt.figure(figsize=(5, 5), dpi=100)
plt.title('Fitted regression plot')
sns.regplot(x=denormalized_targets, y=denormalized_predictions, scatter=True, marker="*", color='orange',line_kws={'color': 'red'})
plt.savefig('result/Fitted regression plot.png')
plt.show()

file_path = 'result/predictions_LSTM_Transformer.csv'
data = pd.read_csv(file_path)
if 'predictions' in data.columns and 'targets' in data.columns:
    correlation = data['predictions'].corr(data['targets'])
    print(f"Predictions and Targets  (COR): {correlation}")
else:
    print("not 'predictions'or 'targets' ")

mae = mean_absolute_error(targets, predictions)
print(f"MAE: {mae:.4f}")  

mape = mean_absolute_percentage_error(targets, predictions)
print(f"MAPE: {mape * 100:.4f}%")  

mse = mean_squared_error(targets, predictions)
print(f"MSE: {mse:.4f}")  

rmse = root_mean_squared_error(targets, predictions)  
print(f"RMSE: {rmse:.4f}")

r2 = r2_score(targets, predictions)
print(f"R²: {r2:.4f}")  
with open('result/metrics.txt', 'w') as f:
    f.write(f"MAE: {mae:.4f}\n")  
    f.write(f"MAPE: {mape * 100:.4f}%\n")  
    f.write(f"MSE: {mse:.4f}\n")  
    f.write(f"RMSE: {rmse:.4f}\n")  
    f.write(f"R²: {r2:.4f}\n") 
    f.write(f"COR: {correlation:.4f}\n") 

print("Metrics have been saved to 'result/metrics.txt'.")
