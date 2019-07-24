#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:21:57 2019

@author: zznj4199
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:25:06 2018

@author: zznj4199
"""
import docx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns



#%% Load all data
def load_data(dir):

    header =  ('Dataset','Method','Architecture','Representation','Embedding','Pretraining','Finetuning','# Questions','run_id', 'MAP','MRR','P@1')
    pd.options.display.float_format = '{:,.2f}'.format
    
    df = pd.read_csv('out/'+dir+'/all_res.csv', sep = ';', names = header)
    mask = (np.logical_and(df.Method !='pointwise',df.Method !='pairwise'))    
    df.loc[mask, '# Questions'] = 0
    mask = (np.logical_and(mask==0,df.Finetuning ==False))    
    df.loc[mask, '# Questions'] = 1
           
    df = df.replace({'# Questions': 'None'}, 1000)  
    df = df.replace({'Embedding': 'wikiqa'}, 'Specific')
    df = df.replace({'Embedding': 'trecqa'}, 'Specific')
    df = df.replace({'Embedding': 'semeval'}, 'Specific')
    df = df.replace({'Embedding': 'yahoo'}, 'Specific')
    df = df.replace({'Embedding': 'insuranceqa'}, 'Specific')
    df = df.replace({'Embedding': 'default'}, 'Generic')      
    #df_s = pd.read_csv('out/'+dir+'/supervised.csv', sep = ';', names =  header)
    
   # df_u = df.loc[]
   # df_s['# Questions']=df_s['# Questions'].replace('None', '1000')
    
    #df = pd.concat([df_u,df_s])
    df['MAP'] = df['MAP']*100
    df.loc[df.Architecture=='default','Architecture']='concat'
    df.loc[df.Architecture=='cos','Architecture']='siamese'
    df = df.reset_index(drop=True)
    df['# Questions'] = df['# Questions'].astype('int')
    return df
#%%
# Unsupervised table
###
dir = ''

df = load_data(dir)
#df = df.loc[df.Finetuning == False]

metrics = ('MAP','MAP','MAP','P@1', 'P@1')
list_res = []
final = pd.DataFrame()

for i,dataset in enumerate(('wikiqa','trecqa','semeval', 'yahoo', 'insuranceqa')):
    tmp = df.loc[df.Dataset == dataset]
    tmp = tmp.loc[tmp.Finetuning == False]
    #tmp = tmp.loc[np.logical_and(tmp.Method=='pairwise',tmp.Architecture =='siamese')]
    
    tmp = tmp.drop(columns=('# Questions'))
    tmp = tmp.reset_index(drop=True)
    if i==0:
        final=tmp.loc[:, ('Architecture','Method', 'Embedding', 'Finetuning')]
        final.reset_index(drop=True)
        
    #tmp = tmp.groupby(['Dataset','Pretraining','Method', 'Architecture','Representation','Embedding']).mean()
    final = pd.concat([final, tmp[metrics[i]]], axis = 1)
    final = final.drop(columns='Finetuning')
    #final.append(tmp[metrics[i]])
    #list_res.append(tmp[metrics[i]].tolist())
    #df = df.sort_index()
    final = final.sort_values(['Architecture','Method','Embedding'])

with open('out/'+dir+'/unsupervised_table.tex', 'a') as tf:
    tf.write('\\begin{table} \n \centering \n')
    tf.write(final.to_latex(index=False))
    tf.write('\\caption{' '} \\end{table} \\\\ \n')
#%%
# Supervised table
###

df = load_data(dir)
df = df.loc[df.Finetuning == True]
df = df.loc[df["# Questions"]==1000]

metrics = ('MAP','MAP','MAP','P@1', 'P@1')
list_res = []
final = pd.DataFrame()

for i,dataset in enumerate(('wikiqa','trecqa','semeval','yahoo', 'insuranceqa')):
    tmp = df.loc[df.Dataset == dataset]
#    tmp = tmp.loc[np.logical_or(np.logical_and(tmp.Method=='pointwise',tmp.Architecture =='concat'),tmp.Pretraining==False)]
    tmp = tmp.drop(columns=('# Questions'))
    tmp = tmp.reset_index(drop=True)
    tmp = tmp.groupby(['Dataset','Architecture','Method', 'Embedding','Pretraining']).mean()
    tmp = tmp.drop(columns=['run_id','Finetuning', 'Representation'])
    tmp.index = tmp.index.droplevel(0)                        
    final = pd.concat([final, tmp[metrics[i]]], axis = 1)
    #final.append(tmp[metrics[i]])
    #list_res.append(tmp[metrics[i]].tolist())
    #df = df.sort_index()

with open('out/'+dir+'/supervised_table.tex', 'a') as tf:
    tf.write('\\begin{table} \n \centering \n')
    tf.write(final.to_latex(index=True))
    tf.write('\\caption{' '} \\end{table} \\\\ \n')


#%% Table distantly supervised
df = load_data(dir)
df_out = pd.DataFrame()   
list_res = []    
for i,dataset in enumerate(('wikiqa','trecqa''semeval','yahoo')):
    df = load_data(dir)
    df = df.loc[df.Dataset == dataset]
    df = df.drop(columns = ('Dataset'))                                                                   
    df = df.replace({'Mode': 'default'}, 'concat')
    df['# Questions'] = df['# Questions'].astype('int')
    df = df.loc[np.logical_and(df.Pretraining==True,df.Finetuning == True)]
    df = df.reset_index(drop=True)                           
    df = df.groupby(['Method','Architecture','Representation','Embedding','Pretraining']).mean()
    df_out = pd.concat((df_out,df))
    index = df.index.values
    list_res.append(df[metrics[i]].tolist())

list_res = list(map(list, zip(*list_res)))    
final = pd.DataFrame(list_res,columns=['WikiQA','TrecQA''SemEval','Yahoo'])
final.index=index

with open('out/'+dir+'dsupervised_table.tex', 'a') as tf:
    tf.write('\\begin{table} \n \centering \n')
    tf.write(final.to_latex(index=True))
    tf.write('\\caption{' '} \\end{table} \\\\ \n')
 

#%% Table supervised
df_out = pd.DataFrame()   
list_res = []    
for i,dataset in enumerate(('wikiqa','semeval','trecqa','yahoo')):
    df = load_data(dir)
    df = df.loc[df.Dataset == dataset]
    df = df.drop(columns = ('Dataset'))                                                                   
    df = df.replace({'Mode': 'default'}, 'concat')
    df['# Questions'] = df['# Questions'].astype('int')
    df = df.loc[np.logical_and(df.Finetuning == True,df['# Questions']==1000)]
    df = df.reset_index(drop=True)                           
    df = df.groupby(['Method','Architecture','Representation','Embedding','Pretraining']).mean()
    df_out = pd.concat((df_out,df))
    index = df.index.values
    list_res.append(df[metrics[i]].tolist())

list_res = list(map(list, zip(*list_res)))    
final = pd.DataFrame(list_res,columns=['WikiQA','SemEval','TrecQA','Yahoo'])
final.index=index

with open('out/'+dir+'supervised_table.tex', 'a') as tf:
    tf.write('\\begin{table} \n \centering \n')
    tf.write(final.to_latex(index=True))
    tf.write('\\caption{' '} \\end{table} \\\\ \n')

#%%

#%% Compare approaches/rch
dir=''
metric = {'wikiqa':'MAP','semeval':'MAP','trecqa':'MAP','yahoo':'P@1', 'insuranceqa':'P@1'}
ranges={'wikiqa':(48,85),'semeval':(60,85),'trecqa':(45,85),'yahoo':(15,40), 'insuranceqa':(0,50)}
df = load_data(dir)
df1 = load_data(dir)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df=df.loc[df["# Questions"] >3]
df = df.loc[df.Representation == 'use']
df = df.loc[df.Pretraining == False]
for i,dataset in enumerate(('wikiqa','trecqa','semeval','yahoo', 'insuranceqa')):

    # Poinwise words
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)        
    ax = sns.lineplot(x="# Questions", y=metric[dataset],hue = 'Method', style = 'Architecture',
                  data=df.loc[np.logical_and(df['# Questions'] >0,df.Dataset == dataset)]).set_title(dataset + ' ')
    
    bl = df1.loc[np.logical_and(df1.Method == 'use', df1.Dataset == dataset)][metric[dataset]].iloc[0]
    ax = sns.lineplot(x = (1,1000), y = bl,color="grey", label="unsupervised", dashes=[6, 2], linestyle='--')
    plt.ylim(ranges[dataset])
    plt.xscale('log', basex=2)
    plt.ylim(ranges[dataset])
    plt.xlim([4,1000])
    ax.set_xticks([4,8,16,32,64,128,256,512,1000])
    ax.set_xticklabels([4,8,16,32,64,128,256,512,'ALL'])
    plt.savefig('out/'+dir+'/'+dataset+'_arch'+'.png', format='png', dpi=200)

#%% Learning representations
ranges={'wikiqa':(68,85),'semeval':(60,85),'trecqa':(45,85),'yahoo':(25,57), 'insuranceqa':(20,50)}
df = load_data(dir)
df1 = load_data(dir)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df=df.loc[df["# Questions"] >3]
#df = df.loc[df.Representation == 'avg_emb']
df = df.loc[df.Embedding != 'default']
df = df.loc[df.Pretraining == False]
architecture = 'siamese'
method = 'pairwise'

# Poinwise words
fig, ax = plt.subplots()
fig.set_size_inches(6,4)        
ax = sns.lineplot(x="# Questions", y="MAP",hue = 'Dataset',style = df.Representation,
              data=df.loc[np.logical_and(np.logical_and(df['# Questions'] >0,df.Architecture == architecture),df.Method == method)]).set_title(dataset + ' ')

bl = df1.loc[df1.Method == 'use']['MAP'].iloc[0]
#ax = sns.lineplot(x = (1,1000), y = bl,color="grey", label="unsupervised", dashes=[6, 2], linestyle='--')
plt.ylim(ranges[dataset])
plt.xscale('log', basex=2)
plt.ylim([42,80])
plt.xlim([4,1000])
ax.set_xticks([4,8,16,32,64,128,256,512,1000])
ax.set_xticklabels([4,8,16,32,64,128,256,512,'ALL'])
plt.savefig('out/'+dir+'/'+dataset+'_'+'_pointw_words'+'.png', format='png', dpi=200)



#%%  Effect of pretraining
dataset = 'wikiqa'
fig, ax = plt.subplots()
fig.set_size_inches(6,4)
df = load_data('') 
rep = 'nn_use'
xticksl=(0,2,8,32,128,512,1024)
df = df.loc[df.Finetuning==True]
sns.lineplot(x="# Questions", y="MAP",hue = 'Dataset',style = 'Pretraining',
              data=df.loc[np.logical_and(df.Representation == 'use',np.logical_and(df.Dataset != 'insuranceqa',
                      np.logical_and(df.Method=='pairwise',df.Architecture=='siamese')))])
#bl = df.loc[np.logical_and(df.Method =='use',np.logical_and( df.Dataset== 'semeval'))]['MAP'].iloc[:].as_matrix()

bl=((70,64,45))
#ax = sns.lineplot(x = (1,1000), y = bl,color="grey", label="unsupervised", dashes=[6, 2], linestyle='--')

sns.regplot(x=np.array([6,6,6]), y=np.array(bl), scatter=True, fit_reg=False, marker='x',
            scatter_kws={"s": 100,'facecolors':sns.color_palette()})
ax.set_xticklabels([5, 10])
plt.ylim(ranges[dataset])
plt.ylim((43,80))
plt.xlim([4,1000])
plt.xscale('log', basex=2)
plt.legend(loc='lower right')
ax.set_xticks([4,8,16,32,64,128,256,512,1000])
ax.set_xticklabels([4,8,16,32,64,128,256,512,'ALL'])
plt.savefig('out/'+dir+'/'+'pretraining.png', format='png', dpi=200)



############################################ Stats DS

pd.options.display.float_format = '{:,.2f}'.format
df = pd.DataFrame(columns = ('Dataset','Split','# questions','# apq','% correct answers','# words question','# words answer')) 

for i,dataset in enumerate(('wikiqa', 'semeval', 'trecqa','yahoo')):
    for j,split in enumerate(('train', 'dev', 'test')):
        qf = open('data/formatted/'+dataset+'/'+split+'_questions')
        af = open('data/formatted/'+dataset+'/'+split+'_answers')
        pf = open('data/formatted/'+dataset+'/'+split+'_pairs')
        
        length = []
        for line in qf:
            length.append(len(line.strip().split()))       
        nb_words_question= np.mean(np.asarray(length))
        nb_questions = len(length)
     
        for line in af:
            length.append(len(line.strip().split()))
        nb_words_answer=np.mean(np.asarray(length))
        nb_answers = len(length)
        pos = 0
        for line in pf:
            line = line.strip().split()
            label = line[2]
            if label == '1':
                pos+=1
        pc_pos =  pos/nb_answers*100
        nb_ApQ = nb_answers/nb_questions                
        df.loc[len(df)]=([dataset,split, nb_questions,nb_ApQ,pc_pos,nb_words_question,nb_words_answer])   
df=df.round(decimals=2)


df = df.groupby(['Dataset']).mean

#%% Table

with open('out/stats_datasets.tex', 'a') as tf:
    tf.write('\\begin{table} \n \centering \n')
    tf.write(df.to_latex(index=True)) 
    tf.write('\caption{Datasets statistics} \n \end{table} \n \\\\ \n')

#%%      
     
import seaborn as sns
tips = sns.load_dataset("tips")
g = sns.catplot(x="Dataset", y="# questions",hue="Split", data=df, kind="bar",height=4, aspect=.7)
