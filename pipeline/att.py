
import torch
from torch import nn

class CondFilterT(nn.Module):
    
    def __init__(self, nevents=5, nconds=1, emb_dim=5, nparams=1):
        super().__init__()
        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)
        self.last_loss = None
        #self.nparams = nparams
        #self.output_size = emb_dim * 2

    def forward(self, input):
        event = input[:,0:1]
        conditions = input[:,1:]

        event_emb = self.event_embedding(event)
        cond_emb = self.event_embedding(conditions)

        event_emb_nrm = event_emb / event_emb.norm(dim=2, keepdim=True)
        cond_emb_nrm = cond_emb / cond_emb.norm(dim=2, keepdim=True)

        scores = event_emb_nrm @ cond_emb_nrm.transpose(1,2)
        self.last_loss = scores.abs().sum()
        print("###", self.last_loss)
        filtered_cond = cond_emb_nrm * scores.transpose(1,2)
        return torch.concat((event_emb.flatten(start_dim=1), filtered_cond.flatten(start_dim=1)), dim=1)
        #norm_scores = torch.softmax(scores)        
        #return scores


class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
        #self.ff = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU()])##, nn.BatchNorm1d(dim)])

    def forward(self, input):
        return self.ff(input) + input


class SimplePred(nn.Module):
    def __init__(self, nevents=5, emb_dim=20, nconds=10, depth=4):
        super().__init__()

        self.nconds = nconds
        self.emb_dim = emb_dim

        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        inner_dim = emb_dim * 2
        self.inner_dim =inner_dim

        self.chain = nn.Sequential(*
            ([Residual(inner_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):
        cond_emb = self.cond_embedding(input[:,1:].int())
        #if self.training:
        #    cond_emb += (torch.rand(cond_emb.shape, device=input.device) - 0.5) / 1000
        bag = cond_emb.sum(dim=1)
        event_emb = self.event_embedding(input[:, 0].int())
        base = torch.concat((event_emb, bag), dim=1)
        return self.chain(base)


class CondFilter(nn.Module):

    def __init__(self, nevents=5, emb_dim=5, inner_dim=10, output_dim=10, depth=8):
        super().__init__()
        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        self.inner = nn.Sequential(*
            ([nn.Linear(emb_dim * 2, inner_dim)] +
            [Residual(inner_dim) for idx in range(depth)] +
            [nn.Linear(inner_dim, output_dim)])
        )
        self.loss = None
        

    def forward(self, input, return_weight=False):
        event = input[:,0]
        conditions = input[:,1]
        event_emb = self.event_embedding(event.int())
        cond_emb = self.cond_embedding(conditions.int())
        emb = torch.concat((event_emb, cond_emb), dim=1)
        out = self.inner(emb)
        return out
        weight = torch.sigmoid(out[:,0:1])
        
        tail = out[:, 1:]

        #norm = tail.norm(dim=1, keepdim=True)
        #norm = torch.where(norm == 0, torch.ones(norm.shape, device=input.device), norm)
        #output = weight * tail  / norm
        #if return_weight:
        #    return weight
        #import pdb
        #pdb.set_trace()
        #if self.loss is None:
        #    self.loss = weight.mean()
        #else:
        #    self.loss += weight.mean()
        return tail


class EventPredictor(nn.Module):

    def __init__(self, nevents=5, emb_dim=5, inner_dim=30, filter_inner_dim=30, depth=5, filter_depth=5):
        super().__init__()
        self.inner_dim = inner_dim
        self.cond_filter = CondFilter(nevents=nevents, emb_dim=emb_dim, 
            inner_dim=filter_inner_dim, output_dim=inner_dim, depth=filter_depth)
        self.chain = nn.Sequential(*
            ([Residual(inner_dim + emb_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim + emb_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):
        base = torch.zeros((input.shape[0], self.inner_dim), device=input.device)        
        for idx in range(1,input.shape[1]):
            base += self.cond_filter(input[:,(0,idx)])
        event = input[:,0]
        event_emb = self.cond_filter.event_embedding(event.int())

        return self.chain(torch.concat((event_emb,base), dim=1))



#model = EventPredictor(nevents=12)
#input = torch.tensor([[5,2,3], [1,3,9], [1,4,5]])
#rv = model(input)
#print(rv)