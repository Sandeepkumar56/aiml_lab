import pandas as pd
import warnings
from pandas import DataFrame
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
df_tennis = pd.read_csv('tennis.csv')

def entropy(probs):
   import math
   return sum( [-prob*math.log(prob, 2) for prob in probs] )

def entropy_of_list(a_list):
   from collections import Counter
   cnt = Counter(x for x in a_list)
   num_instances = len(a_list)*1.0
   probs = [x / num_instances for x in cnt.values()]
   return entropy(probs)

total_entropy = entropy_of_list(df_tennis['PlayTennis'])
print("\n Total Entropy of PlayTennis Data Set:",total_entropy, "\n\n")

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
   df_split = df.groupby(split_attribute_name)
   nobs = len(df.index) * 1.0
   df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
   df_agg_ent.columns = ['Entropy', 'PropObservations']
   new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
   old_entropy = entropy_of_list(df[target_attribute_name])
   return old_entropy - new_entropy

print('\n Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'PlayTennis')),"\n")
print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis, 'Humidity', 'PlayTennis')),"\n")
print('\n Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'PlayTennis')),"\n")
print('\n Info-gain for Temperature is:' + str( information_gain(df_tennis, 'Temperature','PlayTennis')),"\n")

def id3(df, target_attribute_name, attribute_names, default_class=None):
   from collections import Counter
   cnt = Counter(x for x in df[target_attribute_name])
   if len(cnt) == 1:
       return next(iter(cnt))
   elif df.empty or (not attribute_names):
       return default_class
   else:
       default_class = max(cnt.keys())
       gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
       index_of_max = gainz.index(max(gainz))
       best_attr = attribute_names[index_of_max]
       tree = {best_attr:{}}
       remaining_attribute_names = [i for i in attribute_names if i != best_attr]
       for attr_val, data_subset in df.groupby(best_attr):
           subtree = id3(data_subset, target_attribute_name, remaining_attribute_names, default_class)
           tree[best_attr][attr_val] = subtree
       return tree

attribute_names = list(df_tennis.columns)
attribute_names.remove('PlayTennis')
tree = id3(df_tennis,'PlayTennis',attribute_names)

from pprint import pprint
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)

attribute = next(iter(tree))

def classify(instance, tree, default=None):
  attribute = next(iter(tree))
  if instance[attribute] in tree[attribute].keys():
      result = tree[attribute][instance[attribute]]
      if isinstance(result, dict):
          return classify(instance, result)
      else:
          return result
  else:
      return default


df_tennis['actual'] = df_tennis.apply(classify, axis=1, args=(tree,'No'))

training_data = df_tennis.iloc[1:-4]
test_data = df_tennis.iloc[-4:]
train_tree = id3(training_data, 'PlayTennis', attribute_names)
test_data['predicted'] = test_data.apply(classify, axis=1, args=(train_tree,'Yes'))

print("\n\n", test_data)
print ('\n\n Accuracy is : ' + str( sum(test_data['PlayTennis']==test_data['predicted'] ) / (1.0*len(test_data.index)) ))
