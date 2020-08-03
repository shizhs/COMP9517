from pdb import set_trace as bp

def train_net(model, train_dl, loss_fn, optimizer, epochs):
    #for epoch in range(epochs):
    model.train(True)  # Set trainind mode = true
    dataloader = train_dl
    #for x, y in train_dl:
    #for a in train_dl:
    #    x = a[0][0]
    #    y = a[0][1]
    #bp()
    #for x, y in train_dl:
    for a in train_dl:
        #x = x.long()
        #y = y.long()
        x = a[:, 0]
        y = a[:, 1]
        optimizer.zero_grad()
        bp()
        outputs = model(x.unsqueeze(1))
        loss = loss_fn(outputs, y)
        #outputs = model(torch.tensor([[  x ] ]))
        #loss = loss_fn(outputs, torch.tensor([[  y ] ]))

        # the backward pass frees the graph memory, so there is no 
        # need for torch.no_grad in this training pass
        loss.backward()
        optimizer.step()
        print('training')
        # scheduler.step()


