import torch
from dataclasses import dataclass
import pdb

class Sampler(object):

    def __init__(self, events, window=5, nparams=1, nevents=5,
        nconds=1, with_positive=True, num_any=100):

        self.nparams=nparams
        self.nconds = nconds
        self.events = events
        self.tok_no_event = nevents + 1
        self.no_event = torch.tensor([self.tok_no_event] + self.extend_params([]))
        self.window = window
        self.with_positive = with_positive
        self.num_any = num_any

        self.data = torch.stack(
            [torch.tensor([x.logid] + self.extend_params(x.params)) for x in events])
        
        self.times = torch.tensor([x.time for x in events])

        ## helper memeber for randomization
        self.pos = torch.arange(len(self.events))                
        ends = self.times.roll(-1)
        ends[-1] = ends[-2] + window
        ends = torch.min(ends, self.times + window)

        deltas = ends - self.times
        self.cumtime = deltas.cumsum(0)
        self.total_time = deltas.sum()
        
        self.cond_fill = torch.tensor([self.tok_no_event] * nconds)
        

    def rand_time(self, size=2):        
        rnd = torch.rand(size) * self.total_time
        idx = (rnd.unsqueeze(1) >= self.cumtime).sum(dim=1) 
        base = torch.concat((self.cumtime, torch.zeros(1))).roll(1)
        delta = rnd - base[idx]
        times = self.times[idx] + delta
        return times


    def extend_params(self, x):
        if len(x) >= self.nparams:
            return x[0:self.nparams]
        return x + [0] * (self.nparams - len(x))

    def sample(self, num_events, with_positive=None, num_any=None):
        if with_positive is None:
            with_positive = self.with_positive

        if num_any is None:
            num_any = self.num_any

        samples = []
        for idx in range(num_events):
            samples += self.sample_events(with_positive=with_positive, num_any=num_any)
        
        samples= torch.stack(samples, dim=0)
        perm =  torch.randperm(samples.shape[0])
        return samples[perm,...]

    def sample_event(self):
        idx = torch.randint(0, len(self.data), (1,))
        event = self.data[idx,:].squeeze(0)

        if with_positive:
            cond_selection = torch.logical_and(self.times >= self.times[idx] - self.window, self.pos < idx)
            _, conds = self.select_conds(selection)
            samp = torch.concat((event, conds, torch.ones(1)))
            samples.append(samp)        

    def sample_events(self, with_positive=True, num_any=100):
        idx = torch.randint(0, len(self.data), (1,))
        event = self.data[idx,:].squeeze(0)

        samples = self.sample_any(event, num_any)
        if with_positive:
            selection = torch.logical_and(self.times >= self.times[idx] - self.window, self.pos < idx)
            _, conds = self.select_conds(selection)        
            samp = torch.concat((event,conds, torch.ones(1)))
            samples.append(samp)        

        return samples
    
    def sample_any(self, event, size=1):
        if self.nconds > 1:
            return self.sample_any_multi(event, size)
        samps = []
        samp_times = self.rand_time(size).tolist()
        for barrier in samp_times:
            selection = torch.logical_and(self.times >= barrier - self.window *5 , self.times <= barrier)
            idx, conds = self.select_conds(selection)
            future_selection = torch.logical_and(self.times <= barrier + self.window *5, 
                torch.logical_or(self.pos > idx, self.times > barrier))
            target = 0.0
            if (future_selection.sum() > 0):
                ## TODO: should we mask uninteresting params ?
                #target = float(torch.any(torch.all((self.data[future_selection,:] == event), dim=1)))                
                target = torch.all(torch.any(self.data[future_selection,0] == event[0]))
                if future_selection.sum() > 2:
                    pdb.set_trace()
            samps.append(torch.concat((event, conds, torch.tensor([target]))))
            #if (event[0]==2):
            #    pdb.set_trace()

        return samps

    def sample_any(self, event, size=1):
        idx = torch.randint(0, len(self.data), (1,))
        stub = self.data[idx,:].squeeze(0)

        future_selection = torch.logical_and(
            self.pos >= idx, self.times <= self.times[idx])
        matches = (self.data[:,0] == event[0])
        target_crit = torch.logical_and(matches, future_selection)
            if torch.any(target_crit):
                target = 1.0
                target_idx = self.pos[target_crit].min()
                cond_selection = torch.logical_and(
                    self.times >= barrier - self.window * 5,
                    self.pos < target_idx)
            else:
                cond_selection = torch.logical_and(
                    self.times >= barrier - self.window * 5,
                    self.times < barrier)


    def sample_any_multi(self, event, size=1):
        samps = []
        samp_times = self.rand_time(size).tolist()
        for barrier in samp_times:
            future_selection = torch.logical_and(
                self.times >= barrier,
                self.times <= barrier + self.window *5)
            target = 0.0

            matches = (self.data[:,0] == event[0])
            target_crit = torch.logical_and(matches, future_selection)
            if torch.any(target_crit):
                target = 1.0
                target_idx = self.pos[target_crit].min()
                cond_selection = torch.logical_and(
                    self.times >= barrier - self.window * 5,
                    self.pos < target_idx)
            else:
                cond_selection = torch.logical_and(
                    self.times >= barrier - self.window * 5,
                    self.times < barrier)

            
            _, conds = self.select_conds(cond_selection)
            if not torch.any(conds == 0) and event ==3 and target == 1.0:
                pdb.set_trace()
            samps.append(torch.concat((event, conds, torch.tensor([target]))))
        return samps


    def select_conds(self, selection):
        
        if self.nconds > 1:
            return self.times.numel(), torch.concat((self.data[selection,0], self.cond_fill))[0:self.nconds]
        numel = selection.sum()
        if numel == 0:
            return self.times.numel(), torch.concat((self.no_event, torch.ones(1)))

        indexes = self.pos[selection]
        choice = indexes[torch.randint(0,numel, (1,))]
        cond = self.data[choice, :].squeeze(0)
        return choice, torch.concat((cond, torch.tensor([numel])))
        


#sampler = Sampler(SAMPLES)
#sampler.rand_time(3)





