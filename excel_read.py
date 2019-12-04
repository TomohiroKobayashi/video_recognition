import pandas as pd
#スノフラワー0シート
data1 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/スノーフラワー移動距離など.xlsx", sheet_name="スノーフラワー０")
#エクセルの使用する部分をスライシング
data1=data1.ix[13:,:]
#シーン数とラベルを取り出し
df1 = data1.ix[:,[1,8]]
#カラム名を変更
df1 = df1.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

#最後の最小値などのセルを削除
df1 = df1.head(len(df1)-5)

#最後のシーン番号を出力する実験(これでいける)
print(df1.iloc[len(df1)-1, [0]].iat[0])

print("------------------------")
"""
for i in range(len(df1)):
    print(df1.iloc[i, [1]].iat[0])
"""

print(df1[df1['scene']==3009777]['scene'].iat[0])
print(df1[df1['scene']==3009777]['label'].iat[0])
print(type(df1[df1['scene']==3009777]['label'].iat[0]))
label= str(df1[df1['scene']==3009777]['label'].iat[0])
print(label)
print(type(label))
print(df1[df1['scene']==3009777]['label'])
print(df1[df1['scene']==3009777])

print(df1['label'].values.tolist())

print("------------------------")
"""
#スノフラワー1シート
data2 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/スノーフラワー移動距離など.xlsx", sheet_name="スノーフラワー－１")
#エクセルの使用する部分をスライシング
data2=data2.ix[13:,:]
#シーン数とラベルを取り出し
df2 = data2.ix[:,[1,8]]
#カラム名を変更
df2 = df2.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

#最後の最小値などのセルを削除
df2 = df2.head(len(df2)-5)

print("------------------------")

for i in range(len(df2)):
    print(df2.iloc[i, [1]].iat[0])


print("------------------------")
#スノフラワー2シート
data3 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/スノーフラワー移動距離など.xlsx", sheet_name="スノーフラワー－２")
#エクセルの使用する部分をスライシング
data3=data3.ix[13:,:]
#シーン数とラベルを取り出し
df3 = data3.ix[:,[1,8]]
#カラム名を変更
df3 = df3.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

#最後の最小値などのセルを削除
df3 = df3.head(len(df3)-5)

for i in range(len(df3)):
    print(df3.iloc[i, [1]].iat[0])
"""
