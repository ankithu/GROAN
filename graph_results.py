import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style

print(style.available)

style.use('seaborn-v0_8-paper') #sets the size of the charts
style.use('ggplot')

#copied and pasted from terminal results on models/classifier/base_line_model.py
comp_random_micro_f1 =  0.12494056738338
comp_random_f1 = [0.19822829885916515,0.14753397726301393,0.07332213539764844,0.04595669465311533,0.09350486350800126,0.12657413857778255,0.038100301258195995,0.04720624839069608]

dist_micro_f1 = 0.2921443288076496
dist_f1s = [0.4800318283464351,0.18518518518518517,0.05150320354854608,0.030448717948717948,0.08086631349330009,0.11940136054421768,0.025727069351230425,0.03431609473175447]

most_freq_micro_f1 = 0.4805721379893285
most_freq_f1s = [0.64566732,0,0,0, 0, 0, 0,  0]

bert_full_sample_df = pd.read_csv("model_params/classifier/special_tokens_pretrained_bert/model_evaluation_results.csv")
row = bert_full_sample_df.loc[bert_full_sample_df['f1'].idxmax()]
bert_full_sample_df_micro_f1 = row['f1']
bert_full_sample_df_f1 = [row['class_0_f1'], row['class_1_f1'], row['class_2_f1'], row['class_3_f1'], row['class_4_f1'], row['class_5_f1'], row['class_6_f1'], row['class_7_f1']]

bert_under_sample_df = pd.read_csv("model_params/classifier/undersampled_special_tokens_pretrained_bert/model_evaluation_results.csv")
row = bert_under_sample_df.loc[bert_under_sample_df['f1'].idxmax()]
bert_under_sample_df_micro_f1 = row['f1']
bert_under_sample_df_f1 = [row['class_0_f1'], row['class_1_f1'], row['class_2_f1'], row['class_3_f1'], row['class_4_f1'], row['class_5_f1'], row['class_6_f1'], row['class_7_f1']]

few_shot_df = pd.read_csv("zero_shot_results/single-example.csv")
few_shot_tp = few_shot_df['TP'].sum()
few_shot_fp = few_shot_df['FP'].sum()
few_shot_fn = few_shot_df['FN'].sum()
few_shot_micro_precision = few_shot_tp / (few_shot_tp + few_shot_fp + 1e-6)
few_shot_micro_recall = few_shot_tp / (few_shot_tp + few_shot_fn + 1e-6)
few_shot_micro_f1 = 2 * (few_shot_micro_precision * few_shot_micro_recall) / (few_shot_micro_precision + few_shot_micro_recall + 1e-6)
few_shot_f1s = few_shot_df['F1'].tolist()


micro_f1s = [comp_random_micro_f1, dist_micro_f1, most_freq_micro_f1, bert_full_sample_df_micro_f1, bert_under_sample_df_micro_f1, few_shot_micro_f1]

#make a bar plot for micro f1
plt.bar(['Random', 'Distribution', 'Most Frequent', 'BERT Full Sample', 'Bert Under Sample', 'Few Shot'], micro_f1s, color=['b', 'r', 'g', 'c', 'm', 'orange'])
plt.title('F1 (Micro-Average) of Tested Classification Models')
plt.ylabel('F1')
plt.xlabel('Model')
#make x-axis labels vertical
plt.xticks(rotation=10)
plt.figure()

#make a multi-bar plot for f1s
barWidth = 0.15
r1 = range(len(comp_random_f1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
plt.bar(r1, comp_random_f1, color='b', width=barWidth, edgecolor='grey', label='Random Baseline')
plt.bar(r2, dist_f1s, color='r', width=barWidth, edgecolor='grey', label='Distribution Baseline')
plt.bar(r3, most_freq_f1s, color='g', width=barWidth, edgecolor='grey', label='Most Frequent Baseline')
plt.bar(r4, bert_full_sample_df_f1, color='c', width=barWidth, edgecolor='grey', label='BERT-Fintune Full Sample')
plt.bar(r5, bert_under_sample_df_f1, color='m', width=barWidth, edgecolor='grey', label='BERT-Finetune Under Sample')
plt.bar(r6, few_shot_f1s, color='orange', width=barWidth, edgecolor='grey', label='Few Shot')
plt.xlabel('Class')
plt.ylabel('F1')
plt.title('F1 on Each Class of Tested Classification Models')
plt.xticks([r + barWidth for r in range(len(comp_random_f1))], ['Class', 'School', 'Career', 'Club', 'Personal', 'Company', 'News', 'Spam'])
plt.legend(bbox_to_anchor=(0.5, 0.5))
plt.show()
