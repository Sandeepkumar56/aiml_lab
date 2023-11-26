import csv

def g_0(n):
   return ("?",)*n

def s_0(n):
   return ('Φ',)*n

def more_general(h1, h2):
   return all(x == "?" or (x != 'Φ' and (x == y or y == 'Φ')) for x, y in zip(h1, h2))

def fulfills(example, hypothesis):
   return more_general(hypothesis, example)

def min_generalizations(h, x):
   h_new = list(h)
   for i in range(len(h)):
       if not fulfills(x[i:i+1], h[i:i+1]):
           h_new[i] = '?' if h[i] != 'Φ' else x[i]
   return [tuple(h_new)]

def min_specializations(h, domains, x):
   results = []
   for i in range(len(h)):
       if h[i] == "?":
           for val in domains[i]:
               if x[i] != val:
                  h_new = h[:i] + (val,) + h[i+1:]
                  results.append(h_new)
       elif h[i] != 'Φ':
           h_new = h[:i] + ('Φ',) + h[i+1:]
           results.append(h_new)
   return results

with open('weather.csv') as csvFile:
   examples = [tuple(line) for line in csv.reader(csvFile)]

def get_domains(examples):
   d = [set() for _ in examples[0]]
   for x in examples:
       for i, xi in enumerate(x):
           d[i].add(xi)
   return [sorted(x) for x in d]

def candidate_elimination(examples):
   domains = get_domains(examples)[:-1]
   G = {g_0(len(domains))}
   S = {s_0(len(domains))}
   for i, xcx in enumerate(examples):
       x, cx = xcx[:-1], xcx[-1]
       if cx == 'Yes':
           G = {g for g in G if fulfills(x, g)}
           S = generalize_S(x, G, S)
       else:
           S = {s for s in S if not fulfills(x, S)}
           G = specialize_G(x, domains, G, S)
   return G, S

def generalize_S(x, G, S):
   S_prev = list(S)
   for s in S_prev:
       if s not in S:
           continue
       if not fulfills(x, s):
           S.remove(s)
           Splus = min_generalizations(s, x)
           S.update([h for h in Splus if any([more_general(g ,h) for g in G])])
           S.difference_update([h for h in S if any([more_general(h, h1) for h1 in S if h != h1])])
   return S

def specialize_G(x, domains, G, S):
   G_prev = list(G)
   for g in G_prev:
       if g not in G:
           continue
       if fulfills(x, g):
           G.remove(g)
           Gminus = min_specializations(g, domains, x)
           G.update([h for h in Gminus if any([more_general(h, s) for s in S])])
           G.difference_update([h for h in G if any([more_general(g1, h) for g1 in G if h != g1])])
   return G

G, S = candidate_elimination(examples)
