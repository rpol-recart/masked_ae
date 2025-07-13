import pandas as pd

def rolling_normalize(x, window=50):
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-6)

ds=pd.read_pickle('./ds33.pkl')
raw_dataset=[]
metadata=[]
for key in ds.keys():
    reference_ts=ds[key]['TEMP'].values
    if len(reference_ts)<2000:
        continue_
    for sym in ['OBJ1T','OBJ2T','OBJ3T']:
        sample=ds[key][sym].values
        if np.min(sample)>700:
            continue
        raw_dataset.append({'reference':reference_ts,'sample':sample})
        metadata.append({'id':int(key)})

def plot_utils(**kwargs):
  #idxs=[random.randint(0,7200) for _ in range(3)]
  idxs=[442, 3188, 2504]
  df1=pd.DataFrame(raw_dataset[idxs[0]])
  df2=pd.DataFrame(raw_dataset[idxs[1]])
  df3=pd.DataFrame(raw_dataset[idxs[2]])
  
  fig, axs = plt.subplots(3, 1, figsize=(15, 25), sharex=True)
  fig.suptitle(f"Результаты измерений ", fontsize=16)
          
  # Канал 0: ΔT
  axs[0].set_title("Измерение1 : Оригинал образец")
  axs[0].plot(df1.index,df1['sample'], label="Исходный сигнал", color='blue')
  axs[0].plot(df1.index,df1['reference'], label="Исходный сигнал", color='red')
  axs[0].axhline(df1['sample'][(df1['sample']-df1['reference']).idxmin()],c='r',linestyle='--')
  axs[0].legend()
  axs[0].grid(True)
  # Канал 0: ΔT
  axs[1].set_title("Измерение2 : Оригинал образец")
  axs[1].plot(df2.index,df2['sample'], label="Исходный сигнал", color='blue')
  axs[1].plot(df2.index,df2['reference'], label="Исходный сигнал", color='red')
  axs[1].legend()
  axs[1].grid(True)
  # Канал 0: ΔT
  axs[2].set_title("Измерение3 : Оригинал образец")
  axs[2].plot(df3.index,df3['sample'], label="Исходный сигнал", color='blue')
  axs[2].plot(df3.index,df3['reference'], label="Исходный сигнал", color='red')
  axs[2].legend()
  axs[2].grid(True)
  return None

