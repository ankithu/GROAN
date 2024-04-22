import matplotlib.pyplot as plt

comp_random_performance = 0.1306
comp_random_f1 = 0.1618

dist_performance = 0.25323
dist_f1 = 0.25386

most_freq_performance = 0.403
most_freq_f1 = 0.2321

performance = [comp_random_performance, dist_performance, most_freq_performance]
f1 = [comp_random_f1, dist_f1, most_freq_f1]

#make a bar plot for performance
plt.bar(['Random', 'Distribution', 'Most Frequent'], performance)
plt.title('Performance of different baseline models')
plt.ylabel('Performance')
plt.xlabel('Model')
plt.figure()

#make a bar plot for f1
plt.bar(['Random', 'Distribution', 'Most Frequent'], f1)
plt.title('F1 of different baseline models')
plt.ylabel('F1')
plt.xlabel('Model')
plt.show()
