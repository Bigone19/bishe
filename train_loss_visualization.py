import pandas as pd
import matplotlib.pyplot as plt

lines = 10629
result = pd.read_csv('training_loss.log', skiprows=[x for x in range(lines) if ((x % 10 != 9) | (x < 1000))],
                     error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
result.head()

result['loss'] = result['loss'].str.split(' ').str.get(1)
result['avg'] = result['avg'].str.split(' ').str.get(1)
result['rate'] = result['rate'].str.split(' ').str.get(1)
result['seconds'] = result['seconds'].str.split(' ').str.get(1)
result['images'] = result['images'].str.split(' ').str.get(1)
result.head()
result.tail()

print(result['loss'])
print(result['avg'])
print(result['rate'])
print(result['seconds'])
print(result['images'])

result['loss'] = pd.to_numeric(result['loss'])
result['avg'] = pd.to_numeric(result['avg'])
result['rate'] = pd.to_numeric(result['rate'])
result['seconds'] = pd.to_numeric(result['seconds'])
result['images'] = pd.to_numeric(result['images'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['avg'].values, label='avg_loss')
ax.legend(loc='best')
ax.set_title('The loss curves')
ax.set_xlabel('batches')
fig.savefig('avg_loss')
