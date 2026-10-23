"""Persistent logical-slot identities with explicitly authorized seed replacement."""


class CollectionAttempts:
    def __init__(self,entries,events):
        self.entries={s['episode_id']:dict(s) for s in entries}
        if len(self.entries)!=len(entries):raise ValueError('duplicate logical slots')
        self.assignments={key:{slot['seed']:1} for key,slot in self.entries.items()}
        self.used_seeds={s['seed'] for s in entries}
        if len(self.used_seeds)!=len(entries):raise ValueError('original seeds must be globally unique')
        self.failed=set();self.failures=[]
        for row in events:
            if row['event']=='attempt_allocated':self.register(row)
            elif row['event']=='episode_failed':self.record_failure(row)

    def register(self,allocation):
        key=allocation['episode_id'];source=self.entries.get(key)
        if (source is None or allocation.get('target_count')!=source['target_count']
                or allocation.get('source_seed')!=source['seed']):
            raise ValueError('attempt allocation differs from original slot')
        seed=allocation['seed'];attempt=allocation['generation_attempt']
        if type(seed) is not int or not 0<=seed<=4294967295 or type(attempt) is not int:
            raise ValueError('invalid attempt seed or index')
        if seed in self.assignments[key] and self.assignments[key][seed]==attempt:return
        if seed in self.used_seeds or attempt!=max(self.assignments[key].values())+1:
            raise ValueError('attempt seed collision or nonsequential index')
        self.assignments[key][seed]=attempt;self.used_seeds.add(seed)

    def record_failure(self,row):
        key=row['episode_id'];source=self.entries.get(key)
        if (source is None or row.get('target_count')!=source['target_count']
                or row['seed'] not in self.assignments[key]):
            raise ValueError('failure has no recorded seed allocation')
        self.failed.add((key,row['seed']));self.failures.append(row)

    def next_attempt(self,key):
        source=self.entries[key]
        seed,attempt=max(self.assignments[key].items(),key=lambda item:item[1])
        if (key,seed) in self.failed:
            seed=source['seed']+1
            while seed in self.used_seeds:seed+=1
            if seed>4294967295:raise ValueError('seed space exhausted')
            attempt+=1
        return dict(source,seed=seed,source_seed=source['seed'],generation_attempt=attempt)
