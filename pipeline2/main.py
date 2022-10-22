
## replace with import
import logging
from att import SimplePred
import torch
from event import Event, load_event_list
from sampler import Sampler
from model import  train_ext
from att import EventPredictor
from sim import LogType
import logview
from collections import defaultdict
from samples import SAMPLES3

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

if False:
    logs = load_event_list("log.json")
    window = 120
    num_any=20
else:
    logs = SAMPLES3
    window = 100
    num_any = 2
    weight_any = 50
nevents = max([x.logid for x in logs]) + 1
nconds = 10
nparams = 0


logging.info(f"nevents={nevents}; nconds={nconds}; nparms={nparams}")

sampler = Sampler(logs, nevents=nevents, nparams=0, nconds=nconds, window=window, num_any=num_any)
sampler.config(num_any=num_any, weight_any=weight_any, with_positive=False)

sampler.sample_any(torch.tensor([3]),1.0, 20)




#print("####", sampler.sample(1))
#model = EventPredictor(nevents=nevents)
model = SimplePred(nevents=nevents, nconds=nconds)#, depth=6, emb_dim=6)
#model = create_model(nevents=nevents, nparams=nparams, nconds=nconds)
def dep(model, device, event, cond):
    out = model.cond_filter(torch.tensor([[event, cond]], device=device), return_weight=True)
    return (out[0,0].abs().tolist())


def track(idx, model, device):
    print("### C(TASSERT | REJECTED)", dep(model, device, LogType.TICKER_ASSERT, LogType.ORDER_REJECTED))
    print("### C(REJECTED | TASSERT)", dep(model, device, LogType.ORDER_REJECTED, LogType.TICKER_ASSERT))
    print("### C(STOP | TASSERT)", dep(model, device, LogType.STOP_NODES, LogType.TICKER_ASSERT))
    tab = ([[dep(model, device, eventid, condid) for condid in range(nevents)] for eventid in range(nevents)])
    print("\n".join([" ".join(map(lambda x: ("%5s" % f"{(x):.2f}"), ty)) for ty in tab]) )
    #print(tab)
    return True

dlogs = [x.to_dict() for x in logs]

def compile(idx, model, device, window=120):
    model.eval()
    logging.info(f"compiling {idx}")
    lpos = 0
    probs = defaultdict(list)
    causes = defaultdict(list)
    rep = False
    for idx in range(len(logs)):
        while lpos < idx and logs[lpos].time < logs[idx].time - window:
            lpos += 1
        if lpos >= idx:
            continue
        rpos = idx - 1
        bla = ([logs[x].logid for x in range(lpos, idx)])
        conds = ([logs[x].logid for x in range(lpos, idx)] + [sampler.tok_no_event] * sampler.nconds)[0:sampler.nconds]
        
        condidx = [logs[x].logid for x in range(lpos, idx)]

        input = [([logs[idx].logid] + conds) for i in range(idx + 1 - lpos)]
        for ci in range(idx - lpos):
            input[ci + 1][ci + 1] = sampler.tok_no_event
        input = torch.tensor(input)
        values = model(input.to(device)) 
        #print(input, values, sep="\n")
            
        baseval = values[0].tolist()[0]
        for ci in range(idx - lpos):
            probs[idx].append([lpos + ci, baseval])
            altval = values[ci+1].tolist()[0]
            causes[idx].append([lpos + ci, altval])

        if logs[idx].logid == LogType.TICKER_ASSERT and LogType.ORDER_REJECTED in conds and len(bla) == 1 and not rep:
            rep = True
            print(input, values, sep="\n")
    #print("ok")
    logview.compile(dlogs, probs, causes, output = "progress.html")
    return True



def count(aaa):
    ddd=defaultdict(int)
    for idx in aaa:
        ddd[idx] += 1
    return dict(ddd)

train_ext(model, sampler, track=compile, batch_size=200, weight_decay=0.01)
#print(count(torch.round(((sampler.rand_time(100000)+300)/1000)).int().tolist()))





