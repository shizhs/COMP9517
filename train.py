def train_net(model, train_dl, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        model.train(True)  # Set trainind mode = true
        dataloader = train_dl

        for x, y in train_dl:
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # the backward pass frees the graph memory, so there is no 
            # need for torch.no_grad in this training pass
            loss.backward()
            optimizer.step()
            print('training')
            # scheduler.step()



